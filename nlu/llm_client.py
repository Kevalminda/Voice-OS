import logging
import os
import time

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

logger = logging.getLogger(__name__)

# A closed wire schema is easier for Gemini than a free-form entity dictionary.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["create_note", "get_notes", "update_note", "delete_note", "unknown"],
        },
        "entities": {
            "type": "object",
            "properties": {
                "content": {"type": ["string", "null"]},
                "note_id": {"type": ["string", "integer", "null"]},
            },
            "additionalProperties": False,
        },
        "language": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "missing_fields": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["intent", "entities", "language", "confidence", "missing_fields"],
    "additionalProperties": False,
}


def _integer_setting(name, default, low, high):
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        raise ValueError(f"Invalid {name} setting.") from None
    if not low <= value <= high:
        raise ValueError(f"Invalid {name} setting.")
    return value


def get_gemini_client():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing. Check your .env file.")
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=_integer_setting("GEMINI_TIMEOUT_MS", 15000, 100, 120000),
            # Application owns retries; prevent multiplied SDK retries.
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def generate_response(prompt: str) -> str:
    load_dotenv()
    model = os.getenv("GEMINI_MODEL", "").strip()
    if not model:
        raise ValueError("GEMINI_MODEL is missing. Select an available model.")
    attempts = _integer_setting("GEMINI_MAX_ATTEMPTS", 3, 1, 5)
    with get_gemini_client() as client:
        for attempt in range(attempts):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_json_schema=RESPONSE_SCHEMA,
                        temperature=0,
                    ),
                )
                if not isinstance(response.text, str) or not response.text.strip():
                    raise ValueError("Gemini returned an empty response.")
                return response.text
            
            except (errors.APIError, httpx.TransportError) as error:
                code = getattr(error, "code", None)

                retryable = (
                    isinstance(error, httpx.TransportError)
                    or code in {408, 429, 500, 502, 503, 504}
                )

                if not retryable or attempt + 1 == attempts:
                    logger.error(
                        "Gemini request failed after %d attempt(s). "
                        "Error type: %s | HTTP code: %s",
                        attempt + 1,
                        type(error).__name__,
                        code,
                    )
                    raise RuntimeError(
                        f"Gemini request failed. "
                        f"Error type: {type(error).__name__}, "
                        f"HTTP code: {code}. "
                        "Check Gemini availability, quota and model configuration."
                    ) from error

                logger.warning(
                    "Temporary Gemini failure. "
                    "Attempt %d of %d. Error type: %s | HTTP code: %s",
                    attempt + 1,
                    attempts,
                    type(error).__name__,
                    code,
                )

                time.sleep(min(2**attempt, 4))

    raise RuntimeError("Gemini did not return a response.")

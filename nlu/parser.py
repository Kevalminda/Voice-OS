import json

from core.schemas import CommandRequest, Language


def normalize_language(language: str) -> Language:
    if not isinstance(language, str):
        raise ValueError("Language must be a string.")
    value = language.strip().lower().replace("_", "-")
    if value in ("hinglish", "hi-en", "en-hi"):
        return Language.HINGLISH
    return {
        "en": Language.ENGLISH,
        "english": Language.ENGLISH,
        "hi": Language.HINDI,
        "hindi": Language.HINDI,
    }.get(value.split("-")[0], Language.UNKNOWN)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key.")
        result[key] = value
    return result


def parse_nlu_response(response_text: str) -> CommandRequest:
    try:
        data = json.loads(response_text, object_pairs_hook=_unique_object)
        if not isinstance(data, dict):
            raise ValueError("Expected an object.")
        data["language"] = normalize_language(data.get("language", "unknown"))
        return CommandRequest.model_validate(data)
    except (ValueError, TypeError) as error:
        # Do not echo model output, note contents, or SDK exception bodies.
        raise ValueError("Invalid NLU response. Please rephrase your command.") from error

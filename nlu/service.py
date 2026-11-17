from core.schemas import CommandRequest
from nlu.llm_client import generate_response
from nlu.parser import parse_nlu_response


def understand_command(user_text: str) -> CommandRequest:
    if not isinstance(user_text, str) or not user_text.strip():
        raise ValueError("User command cannot be empty.")

    prompt = f"""
    You are the Natural Language Understanding engine for VoiceOS.

    Convert the user's command into structured JSON.

    Supported intents:
    - create_note
    - get_notes
    - update_note
    - delete_note
    - unknown

    Return these fields:
    - intent
    - entities
    - language
    - confidence
    - missing_fields

    Rules:
    - Return valid JSON only.
    - Do not include Markdown.
    - Never invent missing information.
    - If the intent is unclear, use "unknown".
    - Extract only information explicitly provided by the user.

    User command:
    {user_text}
    """

    response_text = generate_response(prompt)

    command = parse_nlu_response(response_text)

    return command

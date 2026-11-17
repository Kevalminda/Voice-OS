from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Intent(str, Enum):
    CREATE_NOTE = "create_note"
    CREATE_REMINDER = "create_reminder"
    WEB_SEARCH = "web_search"
    UNKNOWN = "unknown"
    GET_NOTES = "get_notes"
    DELETE_NOTE = "delete_note"
    UPDATE_NOTE = "update_note"


class Language(str, Enum):
    ENGLISH = "english"
    HINDI = "hindi"
    HINGLISH = "hinglish"
    UNKNOWN = "unknown"


class CommandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Intent
    entities: dict[str, str | None] = Field(default_factory=dict)
    language: Language = Language.UNKNOWN
    confidence: float = Field(ge=0.0, le=1.0, strict=True, allow_inf_nan=False)
    missing_fields: list[str] = Field(default_factory=list)

    @field_validator("entities", mode="before")
    @classmethod
    def normalize_entities(cls, entities):
        if not isinstance(entities, dict):
            raise ValueError("Entities must be an object.")
        normalized = {}
        aliases = {"text": "content", "id": "note_id", "noteid": "note_id"}
        allowed = {"content", "note_id", "task", "time", "query"}
        for name, value in entities.items():
            if not isinstance(name, str):
                raise ValueError("Invalid entity name.")
            name = name.strip().lower().replace("-", "_").replace(" ", "_")
            name = aliases.get(name, name)
            if name not in allowed or name in normalized:
                raise ValueError("Unknown or conflicting entity name.")
            if name == "note_id" and type(value) is int:
                value = str(value)
            if value is not None and not isinstance(value, str):
                raise ValueError("Entity values must be strings (or integer note IDs).")
            normalized[name] = value.strip() if isinstance(value, str) else value
        return normalized


class ActionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["success", "needs_clarification", "rejected", "error", "needs_confirmation"]

    message: str
    data: dict[str, str | int | bool | None] = Field(default_factory=dict)

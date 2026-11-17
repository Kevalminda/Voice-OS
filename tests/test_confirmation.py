"""Offline confirmation tests: controlled tokens, fake time and isolated SQLite only."""

import json
from unittest.mock import MagicMock

import pytest

import main
from actions.notes import delete_note, get_note, update_note
from core.action_router import route_command
from core.confirmation import ConfirmationOutcome, ConfirmationSession
from core.dialogue import DialogueSession
from core.schemas import ActionResult, CommandRequest


def pending(note_id=12, content="Private note content"):
    preview = " ".join(content.split())[:120]
    return ActionResult(
        status="needs_confirmation",
        message=f"Delete note {note_id}: {preview}",
        data={"note_id": note_id, "preview": preview, "original_content": content},
    )


def session(**kwargs):
    return ConfirmationSession(token_factory=lambda: "A7K2", **kwargs)


def delete_command(note_id):
    return CommandRequest(intent="delete_note", entities={"note_id": str(note_id)}, confidence=0.95)


@pytest.mark.parametrize("word", ["yes", "y", "हाँ", "हां", "haan", "han"])
def test_explicit_multilingual_approval_requires_matching_token(word):
    confirmation, request = session(), pending()
    assert confirmation.begin(request) == "A7K2"

    result = confirmation.respond(f"{word} A7K2", request)

    assert result.outcome is ConfirmationOutcome.APPROVED
    assert not confirmation.has_pending


@pytest.mark.parametrize("word", ["no", "n", "nahi", "nahin", "नहीं", "ना"])
def test_multilingual_rejection_closes_request(word):
    confirmation, request = session(), pending()
    confirmation.begin(request)

    assert confirmation.respond(word, request).outcome is ConfirmationOutcome.REJECTED
    assert not confirmation.has_pending


@pytest.mark.parametrize("reply", ["cancel", "/cancel", "रद्द करो", "cancel karo", "CANCEL  KARO"])
def test_multilingual_cancellation_closes_request(reply):
    confirmation, request = session(), pending()
    confirmation.begin(request)

    assert confirmation.respond(reply, request).outcome is ConfirmationOutcome.CANCELLED
    assert not confirmation.has_pending


@pytest.mark.parametrize("reply", [" YES   A7K2 ", "\tHaAn\nA7K2\t", "  हाँ\t A7K2 "])
def test_whitespace_and_phrase_case_are_normalized(reply):
    confirmation, request = session(), pending()
    confirmation.begin(request)
    assert confirmation.respond(reply, request).outcome is ConfirmationOutcome.APPROVED


@pytest.mark.parametrize(
    "reply",
    [
        "yes",
        "हाँ",
        "haan",
        "yes WRONG",
        "yes a7k2",
        "yes Ａ7K2",
        "yes no A7K2",
        "haan nahi A7K2",
        "हाँ नहीं A7K2",
        "yes/no A7K2",
        "yes, A7K2",
        "yesterday A7K2",
        "maybe",
        "",
        "yes A7K2 extra",
        {"approved": True},
        True,
    ],
)
def test_missing_wrong_conflicting_and_unknown_replies_never_approve(reply):
    confirmation, request = session(), pending()
    confirmation.begin(request)

    assert confirmation.respond(reply, request).outcome is ConfirmationOutcome.INVALID
    assert confirmation.has_pending
    assert confirmation.respond("no", request).outcome is ConfirmationOutcome.REJECTED


def test_invalid_reply_budget_cancels_without_extending_deadline():
    now = [0.0]
    confirmation = session(clock=lambda: now[0], max_retries=2)
    request = pending()
    confirmation.begin(request)

    assert confirmation.respond("maybe", request).outcome is ConfirmationOutcome.INVALID
    assert confirmation.remaining_retries == 1
    assert confirmation.respond("yes WRONG", request).outcome is ConfirmationOutcome.CANCELLED
    assert not confirmation.has_pending
    assert confirmation.respond("yes A7K2", request).outcome is ConfirmationOutcome.INVALID


def test_valid_approval_is_consumed_once():
    confirmation, request = session(), pending()
    confirmation.begin(request)

    assert confirmation.respond("yes A7K2", request).outcome is ConfirmationOutcome.APPROVED
    assert confirmation.respond("yes A7K2", request).outcome is ConfirmationOutcome.INVALID
    assert confirmation.token is None


@pytest.mark.parametrize(
    "elapsed,outcome",
    [
        (59.999, ConfirmationOutcome.APPROVED),
        (60.0, ConfirmationOutcome.EXPIRED),
        (600.0, ConfirmationOutcome.EXPIRED),
    ],
)
def test_default_expiry_boundary(elapsed, outcome):
    now = [100.0]
    confirmation = session(clock=lambda: now[0])
    request = pending()
    confirmation.begin(request)
    now[0] += elapsed

    assert confirmation.respond("yes A7K2", request).outcome is outcome
    assert not confirmation.has_pending


def test_invalid_reply_does_not_refresh_expiry():
    now = [0.0]
    confirmation = session(timeout_seconds=10, clock=lambda: now[0])
    request = pending()
    confirmation.begin(request)
    now[0] = 9.0
    assert confirmation.respond("maybe", request).outcome is ConfirmationOutcome.INVALID
    now[0] = 10.0
    assert confirmation.respond("yes A7K2", request).outcome is ConfirmationOutcome.EXPIRED


def test_deadline_is_rechecked_immediately_before_consuming_approval():
    ticks = iter([0.0, 59.9, 60.0])
    confirmation = session(clock=lambda: next(ticks))
    request = pending()
    confirmation.begin(request)

    assert confirmation.respond("yes A7K2", request).outcome is ConfirmationOutcome.EXPIRED
    assert not confirmation.has_pending


def test_old_token_cannot_approve_a_replacement_request():
    tokens = iter(["OLD1", "NEW2"])
    confirmation = ConfirmationSession(token_factory=lambda: next(tokens))
    first, second = pending(12), pending(13)
    confirmation.begin(first)
    confirmation.begin(second)

    assert confirmation.respond("yes OLD1", second).outcome is ConfirmationOutcome.INVALID
    assert confirmation.respond("yes NEW2", second).outcome is ConfirmationOutcome.APPROVED


def test_other_session_token_cannot_approve():
    first, second = session(), ConfirmationSession(token_factory=lambda: "B8L3")
    request = pending()
    first.begin(request)
    second.begin(request)

    assert second.respond("yes A7K2", request).outcome is ConfirmationOutcome.INVALID
    assert first.respond("yes B8L3", request).outcome is ConfirmationOutcome.INVALID
    assert first.respond("yes A7K2", request).outcome is ConfirmationOutcome.APPROVED
    assert second.respond("yes B8L3", request).outcome is ConfirmationOutcome.APPROVED


@pytest.mark.parametrize("change", ["note_id", "content", "preview", "status"])
def test_token_is_bound_to_the_full_original_metadata(change):
    content = "A" * 140 + "original ending"
    request = pending(content=content)
    confirmation = session()
    confirmation.begin(request)
    if change == "note_id":
        request.data["note_id"] = 13
    elif change == "content":
        request.data["original_content"] = "A" * 140 + "different ending"
    elif change == "preview":
        request.data["preview"] = "different preview"
    else:
        request.status = "success"

    assert confirmation.respond("yes A7K2", request).outcome is ConfirmationOutcome.INVALID
    assert not confirmation.has_pending


def test_edit_cancels_without_authorizing_a_later_reply():
    confirmation, request = session(), pending()
    confirmation.begin(request)

    result = confirmation.respond(" EDIT ", request)

    assert result.outcome is ConfirmationOutcome.CANCELLED
    assert result.edit_requested
    assert not confirmation.has_pending
    assert confirmation.respond("yes A7K2", request).outcome is ConfirmationOutcome.INVALID


def test_explicit_cancel_invalidates_current_token():
    confirmation, request = session(), pending()
    confirmation.begin(request)
    assert confirmation.cancel().outcome is ConfirmationOutcome.CANCELLED
    assert confirmation.respond("yes A7K2", request).outcome is ConfirmationOutcome.INVALID


def test_token_factory_is_used_once_per_request_and_cannot_reissue_old_tokens():
    factory = MagicMock(return_value="A7K2")
    confirmation = ConfirmationSession(token_factory=factory)
    request = pending()
    confirmation.begin(request)
    confirmation.respond("no", request)

    with pytest.raises(ValueError):
        confirmation.begin(request)

    assert factory.call_count == 2
    assert not confirmation.has_pending


def test_default_tokens_use_cryptographic_randomness_without_content(monkeypatch):
    random_bytes = MagicMock(side_effect=["1" * 16, "2" * 16])
    monkeypatch.setattr("core.confirmation.secrets.token_hex", random_bytes)
    confirmation = ConfirmationSession()

    first = confirmation.begin(pending(content="SECRET CONTENT"))
    confirmation.cancel()
    second = confirmation.begin(pending(content="SECRET CONTENT"))

    assert first != second
    assert "SECRET" not in first + second
    assert random_bytes.call_count == 2


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"note_id": 12, "preview": "text"},
        {"note_id": True, "preview": "text", "original_content": "text"},
        {"note_id": 12, "preview": "wrong", "original_content": "text"},
    ],
)
def test_incomplete_or_invalid_metadata_cannot_start_a_confirmation(metadata):
    confirmation = session()
    with pytest.raises(ValueError):
        confirmation.begin(
            ActionResult(status="needs_confirmation", message="Delete", data=metadata)
        )
    assert not confirmation.has_pending


@pytest.mark.parametrize(
    "reply",
    ["no", "नहीं", "nahi", "/cancel", "रद्द करो", "edit", "yes WRONG", "yes", "हाँ", "haan"],
)
def test_non_approval_causes_zero_database_mutations(
    monkeypatch,
    isolated_database,
    existing_note,
    reply,
):
    confirmation = session(max_retries=1)
    monkeypatch.setattr(main, "ConfirmationSession", lambda: confirmation)
    monkeypatch.setattr("builtins.input", lambda prompt: reply)
    before = isolated_database.read_bytes()

    result = route_command(delete_command(existing_note), main.confirm_deletion)

    assert result.status == "rejected"
    assert isolated_database.read_bytes() == before
    assert get_note(existing_note).data["content"] == "Test note"
    assert not confirmation.has_pending


def test_main_checks_expiry_after_input_returns(monkeypatch, isolated_database, existing_note):
    now = [0.0]
    confirmation = session(clock=lambda: now[0])
    monkeypatch.setattr(main, "ConfirmationSession", lambda: confirmation)

    def answer(prompt):
        assert "A7K2" in prompt
        now[0] = 60.0
        return "yes A7K2"

    monkeypatch.setattr("builtins.input", answer)
    before = isolated_database.read_bytes()
    assert route_command(delete_command(existing_note), main.confirm_deletion).status == "rejected"
    assert isolated_database.read_bytes() == before
    assert not confirmation.has_pending


def test_main_retries_unknown_input_then_accepts_a_current_token(
    monkeypatch, existing_note, capsys
):
    confirmation = session()
    monkeypatch.setattr(main, "ConfirmationSession", lambda: confirmation)
    answers = iter(["yes", "maybe", "हाँ A7K2"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    result = route_command(delete_command(existing_note), main.confirm_deletion)

    assert result.status == "success"
    assert get_note(existing_note).status == "error"
    output = capsys.readouterr().out
    assert "Delete note" in output
    assert "invalid" in output.lower()
    assert not confirmation.has_pending


def test_changed_note_still_fails_original_atomic_comparison(monkeypatch, existing_note):
    confirmation = session()
    monkeypatch.setattr(main, "ConfirmationSession", lambda: confirmation)

    def answer(prompt):
        update_note(existing_note, "Changed during confirmation")
        return "yes A7K2"

    monkeypatch.setattr("builtins.input", answer)
    assert route_command(delete_command(existing_note), main.confirm_deletion).status == "error"
    assert get_note(existing_note).data["content"] == "Changed during confirmation"


def test_note_deleted_during_confirmation_is_not_deleted_again(monkeypatch, existing_note):
    confirmation = session()
    monkeypatch.setattr(main, "ConfirmationSession", lambda: confirmation)

    def answer(prompt):
        delete_note(existing_note)
        return "yes A7K2"

    monkeypatch.setattr("builtins.input", answer)
    assert route_command(delete_command(existing_note), main.confirm_deletion).status == "error"
    assert get_note(existing_note).status == "error"


def test_corrected_command_requires_a_new_confirmation_token(monkeypatch, existing_note):
    tokens = iter(["OLD1", "NEW2"])
    confirmation = ConfirmationSession(token_factory=lambda: next(tokens))
    monkeypatch.setattr(main, "ConfirmationSession", lambda: confirmation)
    generate = MagicMock(
        return_value=json.dumps(
            {
                "intent": "delete_note",
                "entities": {"note_id": existing_note},
                "language": "english",
                "confidence": 0.95,
                "missing_fields": [],
            }
        )
    )
    monkeypatch.setattr("nlu.service.generate_response", generate)
    dialogue = DialogueSession()
    answers = iter(["edit", "yes OLD1", "yes NEW2"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))

    assert (
        dialogue.process_text_command("Delete a note", main.confirm_deletion).status == "rejected"
    )
    assert get_note(existing_note).status == "success"
    assert not dialogue.has_pending
    assert dialogue.process_text_command("Delete note 1", main.confirm_deletion).status == "success"
    assert get_note(existing_note).status == "error"
    assert generate.call_count == 2


def test_prompt_displays_only_preview_and_exact_token(monkeypatch, capsys):
    confirmation = session()
    monkeypatch.setattr(main, "ConfirmationSession", lambda: confirmation)
    content = "A" * 140 + "DO NOT DISPLAY THIS TAIL"
    prompts = []

    def answer(prompt):
        prompts.append(prompt)
        return "no"

    monkeypatch.setattr("builtins.input", answer)
    assert main.confirm_deletion(pending(content=content)) is False
    assert "A7K2" in prompts[0]
    assert "DO NOT DISPLAY THIS TAIL" not in capsys.readouterr().out + prompts[0]


def test_router_passes_full_original_snapshot_and_does_not_refetch_after_approval(
    monkeypatch,
    existing_note,
):
    from core import action_router

    original_get = action_router.get_note
    get = MagicMock(wraps=original_get)
    monkeypatch.setattr(action_router, "get_note", get)

    def approve(request):
        assert request.data["original_content"] == "Test note"
        return True

    assert route_command(delete_command(existing_note), approve).status == "success"
    get.assert_called_once_with(existing_note)


@pytest.mark.parametrize("signal", [EOFError, KeyboardInterrupt])
def test_input_interruption_clears_confirmation_without_deletion(
    monkeypatch, existing_note, signal
):
    confirmation = session()
    monkeypatch.setattr(main, "ConfirmationSession", lambda: confirmation)
    monkeypatch.setattr("builtins.input", MagicMock(side_effect=signal))

    with pytest.raises(signal):
        route_command(delete_command(existing_note), main.confirm_deletion)

    assert not confirmation.has_pending
    assert get_note(existing_note).data["content"] == "Test note"


@pytest.mark.parametrize(
    "settings",
    [
        {"max_retries": 0},
        {"max_retries": True},
        {"timeout_seconds": 0},
        {"timeout_seconds": float("inf")},
        {"timeout_seconds": float("nan")},
    ],
)
def test_invalid_confirmation_limits_are_rejected(settings):
    with pytest.raises(ValueError):
        ConfirmationSession(**settings)

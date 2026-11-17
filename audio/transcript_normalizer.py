"""Conservative token/phrase spelling corrections, without translation or an LLM."""

import json
import unicodedata
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_VOCABULARY = {
    "VoiceOS": ["Voice OS", "वॉइस ओएस", "वॉइसओएस", "वॉइस ओ एस", "वोई सो इस"],
    "Python": ["पाइथन", "पायथन"],
    "Gemini": ["जेमिनी", "जेमिनाई"],
    "project": ["प्रोजेक्ट"],
    "application": ["एप्लिकेशन", "एप्लीकेशन"],
    "DBMS": ["डीबीएमएस", "डी बी एम एस"],
    "microphone": ["माइक्रोफोन", "माइक्रोफ़ोन"],
    "Faster-Whisper": ["Faster Whisper", "fasterwhisper", "फास्टर व्हिस्पर"],
    "Keval": ["केवल"],
    "Minda": ["मिंडा", "मिन्दा"],
}

# Detection for review only. Never convert, infer, or repair a numeric value.
NUMBER_WORDS = frozenset(
    (
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty "
        "sixty seventy eighty ninety hundred thousand million billion lakh crore "
        "शून्य एक दो तीन चार पाँच पांच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह "
        "सोलह सत्रह अठारह उन्नीस बीस तीस चालीस पचास साठ सत्तर अस्सी नब्बे सौ हजार हज़ार लाख करोड़"
    ).split()
)


def _word_character(char: str) -> bool:
    return char == "_" or unicodedata.category(char)[0] in {"L", "M", "N"}


def _tokens(text: str):
    """Keep combining marks with their word and retain source offsets."""
    spans = []
    start = None
    for index, char in enumerate(text):
        if _word_character(char):
            if start is None:
                start = index
        elif start is not None:
            spans.append((start, index, text[start:index].casefold()))
            start = None
    if start is not None:
        spans.append((start, len(text), text[start:].casefold()))
    return spans


@dataclass(frozen=True)
class NormalizationResult:
    raw_text: str
    normalized_text: str
    corrections: tuple[tuple[str, str], ...]
    review_flags: tuple[dict[str, str], ...] = ()


class TranscriptNormalizer:
    def __init__(
        self,
        vocabulary: Mapping[str, Sequence[str]] | None = None,
        *,
        ambiguous_terms: Collection[str] = ("Keval", "Minda"),
        name_groups: Sequence[Sequence[str]] = (("Keval", "Minda"),),
        name_contexts: Sequence[str] = ("मेरा नाम", "नाम", "my name is", "I am"),
    ):
        vocabulary = DEFAULT_VOCABULARY if vocabulary is None else vocabulary
        self.ambiguous_terms = frozenset(ambiguous_terms)
        self.name_groups = tuple(tuple(group) for group in name_groups)
        self.name_contexts = tuple(tuple(token[2] for token in _tokens(c)) for c in name_contexts)
        self.aliases = {}
        self.separators = {}
        for canonical, candidates in vocabulary.items():
            if not isinstance(canonical, str) or not canonical.strip():
                raise ValueError("Vocabulary canonical terms must be nonempty strings.")
            if not isinstance(candidates, (tuple, list)):
                raise ValueError("Vocabulary aliases must be lists of strings.")
            for alias in [canonical, *candidates]:
                if not isinstance(alias, str) or not alias.strip():
                    raise ValueError("Vocabulary aliases must be nonempty strings.")
                tokens = _tokens(alias)
                if not tokens or tokens[0][0] != 0 or tokens[-1][1] != len(alias):
                    raise ValueError("Vocabulary aliases must contain word phrases only.")
                gaps = tuple(alias[a[1] : b[0]] for a, b in zip(tokens, tokens[1:]))
                if any(not gap.isspace() and gap != "-" for gap in gaps):
                    raise ValueError(
                        "Vocabulary phrases allow whitespace or a literal hyphen only."
                    )
                key = tuple(token[2] for token in tokens)
                if key in self.aliases and self.aliases[key] != canonical:
                    raise ValueError("Conflicting vocabulary aliases.")
                self.aliases[key] = canonical
                self.separators.setdefault(key, set()).add(
                    tuple(" " if gap.isspace() else gap for gap in gaps)
                )
        self.max_tokens = max(map(len, self.aliases), default=0)

    @classmethod
    def from_file(cls, path: str):
        """Custom vocabulary replaces defaults; no external service is consulted."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            not isinstance(data, dict)
            or set(data)
            - {
                "vocabulary",
                "ambiguous_terms",
                "name_groups",
                "name_contexts",
            }
            or not isinstance(data.get("vocabulary"), dict)
        ):
            raise ValueError("Invalid vocabulary configuration.")
        ambiguous = data.get("ambiguous_terms", ["Keval", "Minda"])
        groups = data.get("name_groups", [["Keval", "Minda"]])
        contexts = data.get("name_contexts", ["मेरा नाम", "नाम", "my name is", "I am"])
        if not isinstance(contexts, list) or not all(
            isinstance(item, str) and item.strip() for item in contexts
        ):
            raise ValueError("name_contexts must be a list of nonempty phrases.")
        if not isinstance(ambiguous, list) or not all(isinstance(x, str) for x in ambiguous):
            raise ValueError("ambiguous_terms must be a list of names.")
        if not isinstance(groups, list) or any(
            not isinstance(group, list)
            or len(group) < 2
            or not all(isinstance(x, str) for x in group)
            for group in groups
        ):
            raise ValueError("name_groups must contain lists of at least two names.")
        return cls(
            data["vocabulary"],
            ambiguous_terms=ambiguous,
            name_groups=groups,
            name_contexts=contexts,
        )

    def _name_context(self, text: str, start: int) -> bool:
        prefix = _tokens(text[:start])
        for context in self.name_contexts:
            if not context or len(prefix) < len(context):
                continue
            part = prefix[-len(context) :]
            if (
                tuple(token[2] for token in part) == context
                and all(text[a[1] : b[0]].isspace() for a, b in zip(part, part[1:]))
                and text[part[-1][1] : start].isspace()
            ):
                return True
        return False

    def _review_flags(self, text, tokens, matches, approved_names):
        flags = []
        name_spans = []
        stops = {"and", "is", "with", "from", "है", "हूँ", "हूं", "और", "पर", "में"}
        # Explicit introductions can contain unknown names. Do not guess their spelling.
        for index, token in enumerate(tokens):
            introduction = False
            for context in self.name_contexts:
                # "I am testing" is not sufficient evidence of a name introduction.
                if not context or context == ("i", "am") or index < len(context):
                    continue
                prefix = tokens[index - len(context) : index]
                if (
                    tuple(part[2] for part in prefix) == context
                    and all(text[a[1] : b[0]].isspace() for a, b in zip(prefix, prefix[1:]))
                    and text[prefix[-1][1] : token[0]].isspace()
                ):
                    introduction = True
                    break
            if not introduction:
                continue
            end = token[0]
            for offset in range(index, min(index + 4, len(tokens))):
                start, finish, word = tokens[offset]
                if word in stops or (end != token[0] and not text[end:start].isspace()):
                    break
                end = finish
            if end > token[0]:
                name_spans.append((token[0], end))
        for index, (start, end, canonical) in enumerate(matches):
            if canonical in self.ambiguous_terms and index not in approved_names:
                if not any(left <= start and end <= right for left, right in name_spans):
                    name_spans.append((start, end))
        for start, end in sorted(set(name_spans)):
            flags.append(
                {
                    "kind": "name",
                    "text": text[start:end],
                    "message": "Verify any name or ambiguous term here; spelling may be uncertain.",
                }
            )
        numeric = [
            word in NUMBER_WORDS or any(c.isdecimal() for c in word) for _, _, word in tokens
        ]
        index = 0
        while index < len(tokens):
            if not numeric[index]:
                index += 1
                continue
            start, end, _ = tokens[index]
            index += 1
            while index < len(tokens) and numeric[index]:
                gap = text[end : tokens[index][0]]
                if not gap.isspace() and gap not in {"-", ".", ",", "/", ":"}:
                    break
                end = tokens[index][1]
                index += 1
            flags.append(
                {
                    "kind": "number",
                    "text": text[start:end],
                    "message": "Verify this number against the speech; its value is unchanged.",
                }
            )
        return tuple(flags)

    def normalize(self, raw_text: str) -> NormalizationResult:
        tokens = _tokens(raw_text)
        matches = []
        index = 0
        while index < len(tokens):
            for size in range(min(self.max_tokens, len(tokens) - index), 0, -1):
                part = tokens[index : index + size]
                key = tuple(token[2] for token in part)
                gaps = tuple(raw_text[a[1] : b[0]] for a, b in zip(part, part[1:]))
                separators = tuple(" " if gap.isspace() else gap for gap in gaps)
                canonical = self.aliases.get(key)
                if separators not in self.separators.get(key, set()):
                    continue
                if canonical is not None:
                    matches.append((part[0][0], part[-1][1], canonical))
                    index += size
                    break
            else:
                index += 1
        # Ambiguous name candidates need the full configured adjacent name, in order.
        approved_names = set()
        for offset in range(len(matches)):
            for group in self.name_groups:
                part = matches[offset : offset + len(group)]
                if (
                    tuple(item[2] for item in part) == group
                    and self._name_context(raw_text, part[0][0])
                    and all(raw_text[a[1] : b[0]].isspace() for a, b in zip(part, part[1:]))
                ):
                    approved_names.update(range(offset, offset + len(group)))
        parts, corrections, cursor = [], [], 0
        for index, (start, end, canonical) in enumerate(matches):
            if canonical in self.ambiguous_terms and index not in approved_names:
                continue
            source = raw_text[start:end]
            if source != canonical:
                parts.extend((raw_text[cursor:start], canonical))
                corrections.append((source, canonical))
                cursor = end
        parts.append(raw_text[cursor:])
        return NormalizationResult(
            raw_text,
            "".join(parts),
            tuple(corrections),
            self._review_flags(raw_text, tokens, matches, approved_names),
        )

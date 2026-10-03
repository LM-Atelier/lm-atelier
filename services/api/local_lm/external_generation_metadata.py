"""The generation settings a picture made elsewhere carries in its own file, read as plain claims.

Only text a PNG stores uncompressed is read (png_text_metadata). Two ways of
writing settings are recognized:

- ``parameters``: the text most image tools write: the prompt, then a line that
  starts ``Negative prompt:``, then one line of ``Key: value`` pairs.
- ``prompt``: a ComfyUI API graph, read for its one sampler and the nodes that
  sampler names for its prompts and its empty picture.

Every claim is one plain value with the name it was found under. Nothing is
kept, fetched, run or trusted here: a model, a LoRA or a file named in the text
is listed as ignored, never looked up by name, and a graph is never imported.
Each value is bounded, and text that could change how other text is shown is
left out rather than shown.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Final

from .png_text_metadata import MAX_STRING_BYTES, PngTextClaim, read_png_text

PARSER_VERSION: Final = 1


@dataclass(frozen=True)
class MetadataParseBudget:
    """How much of a picture's metadata is read: one version for every reader."""

    version: int
    max_json_depth: int
    max_json_nodes: int
    max_claims: int
    max_string_bytes: int


BUDGET: Final = MetadataParseBudget(
    version=1,
    max_json_depth=32,
    max_json_nodes=10_000,
    max_claims=256,
    max_string_bytes=MAX_STRING_BYTES,
)

#: The settings a claim can name.
CLAIM_KEYS: Final = (
    "prompt",
    "negative_prompt",
    "seed",
    "steps",
    "guidance",
    "sampler",
    "scheduler",
    "denoise",
    "width",
    "height",
    "batch",
)

_SETTINGS_LINE = re.compile(r"^Steps: ", re.MULTILINE)
#: The longest settings line read; real ones are a few kilobytes at most.
_MAX_SETTINGS_LINE = 16 * 1024
_MAX_SETTING_NAME = 64
_SIZE = re.compile(r"(\d{1,5})x(\d{1,5})")
_NEGATIVE = "Negative prompt:"
#: Settings-line names read, with the claim each one is and how it is read.
_PARAMETERS: Final = {
    "Steps": ("steps", "count"),
    "Sampler": ("sampler", "name"),
    "Schedule type": ("scheduler", "name"),
    "CFG scale": ("guidance", "number"),
    "Seed": ("seed", "seed"),
    "Denoising strength": ("denoise", "fraction"),
    "Batch size": ("batch", "count"),
}
#: Names that point at files on someone else's computer: never followed.
_NAMED_FILES = re.compile(
    r"(model|vae|lora|lyco|embedding|hypernet|checkpoint|ckpt|unet|clip_name)", re.IGNORECASE
)
_MODEL_FILE = re.compile(r"\.(safetensors|ckpt|pt|pth|bin|gguf|sft)$", re.IGNORECASE)
#: Sampler inputs read through the node they link to rather than as values.
_LINKED_INPUTS = frozenset({"positive", "negative", "latent_image"})
_SAMPLERS: Final = {
    "KSampler": {
        "seed": ("seed", "seed"),
        "steps": ("steps", "count"),
        "cfg": ("guidance", "number"),
        "sampler_name": ("sampler", "name"),
        "scheduler": ("scheduler", "name"),
        "denoise": ("denoise", "fraction"),
    },
    "KSamplerAdvanced": {
        "noise_seed": ("seed", "seed"),
        "steps": ("steps", "count"),
        "cfg": ("guidance", "number"),
        "sampler_name": ("sampler", "name"),
        "scheduler": ("scheduler", "name"),
    },
}
_EMPTY_PICTURES = frozenset({"EmptyLatentImage", "EmptySD3LatentImage"})
_EMPTY_PICTURE_INPUTS: Final = {
    "width": ("width", "size"),
    "height": ("height", "size"),
    "batch_size": ("batch", "count"),
}


@dataclass(frozen=True)
class RemixClaim:
    """One setting the picture's file states, and the name it was found under."""

    key: str
    value: str | int | float
    source: str


@dataclass(frozen=True)
class IgnoredMetadata:
    """Something in the picture's file that was read and deliberately not used, and why."""

    name: str
    reason: str


@dataclass(frozen=True)
class ExternalGenerationMetadata:
    """What a picture's own file says about how it was made, in plain claims."""

    dialect: str
    parser_version: int
    budget_version: int
    #: The digest of the text read, so two readings can be compared; never the text.
    digest: str | None
    claims: tuple[RemixClaim, ...]
    ignored: tuple[IgnoredMetadata, ...]
    warnings: tuple[str, ...]


def read_external_generation_metadata(payload: bytes) -> ExternalGenerationMetadata:
    """Read a PNG's settings text as claims; a file that is not a PNG yields none.

    Raises ValueError for a PNG whose text chunks are damaged or past the
    reader's ceilings; nothing is returned from such a file.
    """

    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return metadata_not_read("format_not_read")
    text = read_png_text(payload)
    reader = _Reader()
    for skipped in text.skipped:
        reader.ignore(skipped.keyword or "text", skipped.reason)
    by_keyword: dict[str, PngTextClaim] = {}
    for claim in text.claims:
        if claim.keyword in by_keyword:
            reader.ignore(claim.keyword, "repeated")
            continue
        by_keyword[claim.keyword] = claim
    dialect = "none"
    if "parameters" in by_keyword:
        dialect = "parameters"
        reader.parameters(by_keyword.pop("parameters").text)
    elif "prompt" in by_keyword:
        dialect = "comfyui_prompt"
        reader.comfyui_prompt(by_keyword.pop("prompt").text)
    for keyword in sorted(by_keyword):
        # Both a workflow and an unread prompt graph are graphs, never imported.
        reason = "workflow_graph" if keyword in {"workflow", "prompt"} else "not_settings"
        reader.ignore(keyword, reason)
    digest = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                [[claim.keyword, claim.text] for claim in text.claims],
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest()
        if text.claims
        else None
    )
    warnings = tuple(dict.fromkeys(reader.warnings))
    if dialect == "none":
        warnings = (*warnings, "no_settings_found")
    return ExternalGenerationMetadata(
        dialect=dialect,
        parser_version=PARSER_VERSION,
        budget_version=BUDGET.version,
        digest=digest,
        claims=tuple(reader.claims),
        ignored=tuple(reader.ignored),
        warnings=warnings,
    )


def metadata_not_read(warning: str) -> ExternalGenerationMetadata:
    """The answer for a file whose metadata is not read at all, and why."""

    return ExternalGenerationMetadata(
        dialect="none",
        parser_version=PARSER_VERSION,
        budget_version=BUDGET.version,
        digest=None,
        claims=(),
        ignored=(),
        warnings=(warning,),
    )


class _Reader:
    def __init__(self) -> None:
        self.claims: list[RemixClaim] = []
        self.ignored: list[IgnoredMetadata] = []
        self.warnings: list[str] = []
        self._seen: set[str] = set()

    def ignore(self, name: str, reason: str) -> None:
        entry = IgnoredMetadata(_shown_name(name), reason)
        if entry in self.ignored:
            return
        if len(self.ignored) < BUDGET.max_claims:
            self.ignored.append(entry)
        else:
            self.warnings.append("too_many_entries")

    def claim(self, key: str, raw: object, kind: str, source: str) -> None:
        if key in self._seen:
            self.ignore(source, "repeated")
            return
        if kind == "text" and isinstance(raw, str) and not raw.strip():
            # An empty prompt is a real answer, common for a negative one.
            self.ignore(source, "empty")
            return
        value = _value(raw, kind)
        if value is None:
            self.ignore(source, "malformed")
            return
        if isinstance(value, str) and not _shown_safely(value):
            self.ignore(source, "unsafe_text")
            return
        if len(self.claims) >= BUDGET.max_claims:
            self.warnings.append("too_many_entries")
            return
        self._seen.add(key)
        self.claims.append(RemixClaim(key, value, source))

    # ----- the settings text most image tools write ------------------------------

    def parameters(self, text: str) -> None:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        settings = list(_SETTINGS_LINE.finditer(text))
        if not settings:
            self.warnings.append("settings_unrecognized")
            self.ignore("parameters", "settings_unrecognized")
            return
        start = settings[-1].start()
        head, line = text[:start].rstrip("\n"), text[start:].split("\n", 1)
        if len(line) > 1 and line[1].strip():
            # Lines after the settings line belong to no setting this reads.
            self.ignore("parameters", "trailing_text")
        negative_at = head.rfind("\n" + _NEGATIVE)
        if head.startswith(_NEGATIVE):
            negative_at = 0
        prompt, negative = head, None
        if negative_at >= 0:
            prompt = head[:negative_at]
            negative = head[negative_at:].lstrip("\n")[len(_NEGATIVE) :].strip()
        if prompt.strip():
            self.claim("prompt", prompt.strip(), "text", "prompt")
        if negative is not None:
            self.claim("negative_prompt", negative, "text", "Negative prompt")
        if len(line[0]) > _MAX_SETTINGS_LINE:
            self.ignore("parameters", "too_long")
            return
        self._settings_line(line[0])

    def _settings_line(self, line: str) -> None:
        for piece in _settings_pieces(line):
            name, colon, value = piece.partition(":")
            name, raw = name.strip(), value.strip()
            if not name and not raw:
                continue
            if not colon:
                # A setting written as its name alone, the way a switch is.
                self.ignore(name, "not_used")
                continue
            if not name or len(name) > _MAX_SETTING_NAME:
                self.ignore("parameters", "malformed")
                continue
            if len(raw) > 1 and raw.startswith('"') and raw.endswith('"'):
                # A value the writer quoted; a quote anywhere else is part of the value.
                try:
                    decoded = json.loads(raw)
                except ValueError:
                    decoded = None
                if not isinstance(decoded, str):
                    self.ignore(name, "malformed")
                    continue
                raw = decoded
            if name == "Size":
                size = _SIZE.fullmatch(str(raw))
                if size is None:
                    self.ignore(name, "malformed")
                    continue
                self.claim("width", int(size.group(1)), "size", "Size")
                self.claim("height", int(size.group(2)), "size", "Size")
            elif name in _PARAMETERS:
                key, kind = _PARAMETERS[name]
                self.claim(key, raw, kind, name)
            elif _NAMED_FILES.search(name):
                # A file named by its name or a short hash is not an identity here.
                self.ignore(name, "names_a_file")
            else:
                self.ignore(name, "not_used")

    # ----- a ComfyUI API graph ------------------------------------------------------

    def comfyui_prompt(self, text: str) -> None:
        try:
            graph = json.loads(text, object_pairs_hook=_no_repeated_keys)
        except (ValueError, RecursionError):
            self.ignore("prompt", "malformed")
            return
        if not _within_budget(graph):
            self.ignore("prompt", "too_complex")
            return
        if not isinstance(graph, dict) or not all(
            isinstance(node, dict) and isinstance(node.get("inputs", {}), dict)
            for node in graph.values()
        ):
            self.ignore("prompt", "malformed")
            return
        samplers = [(node_id, node) for node_id, node in graph.items() if _class(node) in _SAMPLERS]
        if len(samplers) != 1:
            reason = "no_sampler" if not samplers else "several_samplers"
            self.warnings.append(reason)
            self.ignore("prompt", reason)
            return
        sampler_id, sampler = samplers[0]
        kind = _class(sampler)
        inputs = sampler.get("inputs") or {}
        for name, raw in inputs.items():
            if name in _SAMPLERS[kind]:
                key, value_kind = _SAMPLERS[kind][name]
                self._node_value(key, raw, value_kind, f"{kind}.{name}")
            elif name not in _LINKED_INPUTS:
                self.ignore(f"{kind}.{name}", _input_reason(name, raw))
        read = {sampler_id}
        read |= self._linked_text(graph, inputs, "positive", "prompt", kind)
        read |= self._linked_text(graph, inputs, "negative", "negative_prompt", kind)
        if "latent_image" in inputs:
            picture_id = _link_target(inputs["latent_image"])
            picture = graph.get(picture_id) if picture_id is not None else None
            if picture_id is None or not isinstance(picture, dict):
                self.ignore(f"{kind}.latent_image", "malformed")
            elif _class(picture) in _EMPTY_PICTURES:
                # Any other kind of starting picture is listed with the other nodes.
                read.add(picture_id)
                picture_kind = _class(picture)
                picture_inputs = picture.get("inputs") or {}
                for name, (key, value_kind) in _EMPTY_PICTURE_INPUTS.items():
                    if name in picture_inputs:
                        self._node_value(
                            key, picture_inputs[name], value_kind, f"{picture_kind}.{name}"
                        )
                self._other_inputs(picture_kind, picture_inputs, set(_EMPTY_PICTURE_INPUTS))
        for node_id, node in graph.items():
            if node_id not in read:
                # Every other node is listed by its class, never run or imported.
                reason = "names_a_file" if _names_a_file(node) else "not_used"
                self.ignore(_class(node) or "node", reason)

    def _node_value(self, key: str, raw: object, kind: str, source: str) -> None:
        if isinstance(raw, list):
            # The value comes from another node, which this does not follow.
            self.ignore(source, "from_another_node")
            return
        self.claim(key, raw, kind, source)

    def _linked_text(
        self, graph: dict[str, Any], inputs: dict[str, Any], name: str, key: str, kind: str
    ) -> set[str]:
        """Claim the text a sampler's prompt input links to; the node read, if any."""

        if name not in inputs:
            return set()
        source = f"{kind}.{name}"
        node_id = _link_target(inputs[name])
        node = graph.get(node_id) if node_id is not None else None
        if node_id is None or not isinstance(node, dict):
            self.ignore(source, "malformed")
            return set()
        if _class(node) != "CLIPTextEncode":
            # That node is listed by its class with the other nodes.
            self.ignore(source, "not_plain_text")
            return set()
        node_inputs = node.get("inputs") or {}
        text = node_inputs.get("text")
        if isinstance(text, list):
            self.ignore(source, "from_another_node")
        else:
            self.claim(key, text, "text", source)
        self._other_inputs("CLIPTextEncode", node_inputs, {"text"})
        return {node_id}

    def _other_inputs(self, kind: str, inputs: dict[str, Any], read: set[str]) -> None:
        """List the inputs of a node that was read, other than the ones claimed."""

        for name, value in inputs.items():
            if name not in read:
                self.ignore(f"{kind}.{name}", _input_reason(name, value))


def _settings_pieces(line: str) -> list[str]:
    """A settings line split at the commas outside quoted values, in linear time.

    The writer quotes a value only when it must, so a quote opens a value only
    right after its name's colon, and only when it closes just before a comma
    or the end of the line; any other quote is part of the value.
    """

    closing = _closing_quotes(line)
    pieces: list[str] = []
    start = index = 0
    named = False
    while index < len(line):
        character = line[index]
        if character == ",":
            pieces.append(line[start:index])
            start = index + 1
            named = False
        elif character == ":" and not named:
            named = True
            value = index + 1
            while value < len(line) and line[value] == " ":
                value += 1
            end = closing[value + 1] if value < len(line) and line[value] == '"' else -1
            if end >= 0:
                after = end + 1
                while after < len(line) and line[after] == " ":
                    after += 1
                if after == len(line) or line[after] == ",":
                    index = after
                    continue
        index += 1
    pieces.append(line[start:])
    return pieces


def _closing_quotes(line: str) -> list[int]:
    """Where a quoted value read from each position would close, or -1 when it never does.

    Read from the end, so the whole line costs one pass: inside a quoted value a
    backslash escapes the next character.
    """

    closing = [-1] * (len(line) + 2)
    for index in range(len(line) - 1, -1, -1):
        character = line[index]
        if character == '"':
            closing[index] = index
        elif character == "\\":
            closing[index] = closing[index + 2]
        else:
            closing[index] = closing[index + 1]
    return closing


def _class(node: dict[str, Any]) -> str:
    """A node's class name, or nothing when it is not a plain name."""

    value = node.get("class_type")
    return value if isinstance(value, str) else ""


def _link_target(link: object) -> str | None:
    """The node a ``[node id, output]`` link names, or nothing when it is not a link."""

    if (
        isinstance(link, list)
        and len(link) == 2
        and isinstance(link[0], str)
        and type(link[1]) is int
    ):
        return link[0]
    return None


def _names_a_file(node: dict[str, Any]) -> bool:
    inputs = node.get("inputs") or {}
    return any(_input_reason(name, value) == "names_a_file" for name, value in inputs.items())


def _input_reason(name: str, value: object) -> str:
    """Why one input of a node is left out: whether it names a file, or is just not used."""

    if _link_target(value) is not None:
        # A link to another node names no file itself.
        return "not_used"
    if _NAMED_FILES.search(name) or any(_MODEL_FILE.search(text) for text in _strings(value)):
        return "names_a_file"
    return "not_used"


def _strings(value: object) -> list[str]:
    """Every string in a value, however it is nested; graphs are already within the budget."""

    found: list[str] = []
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            found.append(item)
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return found


def _no_repeated_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("repeated key")
        value[key] = item
    return value


def _within_budget(value: object) -> bool:
    """Whether a parsed value stays inside the depth, node and string ceilings."""

    nodes = 0
    stack: list[tuple[object, int]] = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > BUDGET.max_json_nodes or depth > BUDGET.max_json_depth:
            return False
        if isinstance(item, dict):
            stack.extend((child, depth + 1) for child in item.values())
            stack.extend((key, depth + 1) for key in item)
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, str) and _utf8_size(item) > BUDGET.max_string_bytes:
            return False
    return True


def _value(raw: object, kind: str) -> str | int | float | None:
    """A claim's value read the way its kind is, or None when it is not that kind."""

    if kind == "text":
        if not isinstance(raw, str) or not raw.strip():
            return None
        # Windows line ends read as plain ones, wherever the text was written.
        raw = raw.replace("\r\n", "\n").replace("\r", "\n")
        return raw if _utf8_size(raw) <= BUDGET.max_string_bytes else None
    if kind == "name":
        text = raw.strip() if isinstance(raw, str) else ""
        return text if 0 < len(text) <= 100 else None
    number = _number(raw)
    if number is None:
        return None
    if kind in {"seed", "count", "size"}:
        # Whole numbers are read exactly; a seed can be larger than a float holds.
        if not isinstance(number, int):
            return None
        bounds = {"seed": (0, 2**64 - 1), "count": (1, 10_000), "size": (1, 65_535)}[kind]
        return number if bounds[0] <= number <= bounds[1] else None
    if kind == "fraction":
        return float(number) if 0 <= number <= 1 else None
    return float(number) if -1_000 <= number <= 1_000 else None


def _number(raw: object) -> int | float | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return raw if math.isfinite(raw) else None
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if re.fullmatch(r"-?\d{1,20}", text):
        return int(text)
    # The shortest text that round-trips a float can run to 17 digits or an exponent.
    if re.fullmatch(r"-?\d{1,20}(\.\d{1,20})?([eE][-+]?\d{1,3})?", text):
        number = float(text)
        return number if math.isfinite(number) else None
    return None


def _utf8_size(text: str) -> int:
    """The bytes text takes; a lone surrogate from a JSON escape is measured, not refused."""

    return len(text.encode("utf-8", "surrogatepass"))


def _shown_safely(text: str) -> bool:
    """Whether text holds nothing that would change how it, or text beside it, is shown."""

    for character in text:
        if character in "\n\t":
            continue
        category = unicodedata.category(character)
        if category in {"Cc", "Cf", "Cs", "Co", "Cn"}:
            return False
    return True


def _shown_name(name: str) -> str:
    """A name from the file, bounded and safe to show; replaced when it is not."""

    plain = name.strip()[:100]
    return plain if plain and _shown_safely(plain) else "text"

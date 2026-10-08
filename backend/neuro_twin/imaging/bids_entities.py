"""BIDS filename/entity parsing for auditable MRI provenance.

This parser intentionally handles entity extraction rather than attempting to be a
full BIDS validator. It rejects duplicate entities and preserves the suffix and
extension needed to bind an image to a subject/session/modality record.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


class BIDSEntityError(ValueError):
    """Raised when a filename cannot be interpreted under the supported BIDS grammar."""


# Core MRI entities plus the common derivative-space entities used by this project.
_ENTITY_KEYS = (
    "sub", "ses", "task", "acq", "ce", "rec", "dir", "run", "mod", "echo",
    "flip", "inv", "mt", "part", "chunk", "space", "desc", "label",
)
_TOKEN = re.compile(r"^(?P<key>[A-Za-z0-9]+)-(?P<value>[^_]+)$")


@dataclass(frozen=True)
class BIDSFileRecord:
    path: str
    entities: dict[str, str]
    suffix: str
    extension: str

    @property
    def subject_id(self) -> str:
        value = self.entities.get("sub")
        if not value:
            raise BIDSEntityError("BIDS image is missing the required sub-<label> entity")
        return value

    @property
    def session_id(self) -> str | None:
        return self.entities.get("ses")

    @property
    def space(self) -> str | None:
        return self.entities.get("space")


def _split_extension(name: str) -> tuple[str, str]:
    if name.endswith(".nii.gz"):
        return name[:-7], ".nii.gz"
    if name.endswith(".nii"):
        return name[:-4], ".nii"
    raise BIDSEntityError(f"unsupported imaging extension: {name}")


def parse_bids_image(path: str | Path) -> BIDSFileRecord:
    path = Path(path)
    stem, extension = _split_extension(path.name)
    tokens = stem.split("_")
    if len(tokens) < 2:
        raise BIDSEntityError(f"BIDS image filename must contain entities and a suffix: {path.name}")

    suffix = tokens[-1]
    entities: dict[str, str] = {}
    for token in tokens[:-1]:
        match = _TOKEN.match(token)
        if not match:
            raise BIDSEntityError(f"invalid BIDS entity token {token!r}")
        key = match.group("key")
        value = match.group("value")
        if key not in _ENTITY_KEYS:
            # Preserve unknown entities for provenance rather than silently dropping them.
            entities[key] = value
            continue
        if key in entities:
            raise BIDSEntityError(f"duplicate BIDS entity {key!r}")
        entities[key] = value

    # BIDS raw imaging must have sub. Derivatives may have additional source entities,
    # but retaining the subject requirement here prevents ambiguous subject binding.
    if "sub" not in entities:
        raise BIDSEntityError("BIDS image is missing the required sub-<label> entity")

    return BIDSFileRecord(
        path=str(path),
        entities=entities,
        suffix=suffix,
        extension=extension,
    )

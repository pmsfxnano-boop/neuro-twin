"""BIDS PET blood-processing contracts for AIF ingestion (Batch 18)."""
from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .kinetics import ArterialInputFunction, PETKineticError

BLOOD_CONTRACT_VERSION = "0.1.0-b18"


@dataclass(frozen=True)
class BloodProcessedTable:
    time_s: np.ndarray
    plasma_radioactivity: np.ndarray | None
    metabolite_parent_fraction: np.ndarray | None
    input_function: np.ndarray | None
    columns: tuple[str, ...]
    metadata: dict[str, Any]

    def to_aif(self, *, time_unit: str = "min", delay: float = 0.0) -> ArterialInputFunction:
        if self.plasma_radioactivity is None or self.metabolite_parent_fraction is None:
            raise PETKineticError("bloodproc must contain plasma_radioactivity and metabolite_parent_fraction to build an AIF")
        if np.any(self.plasma_radioactivity < 0):
            raise PETKineticError("plasma_radioactivity must be non-negative")
        if np.any((self.metabolite_parent_fraction < 0) | (self.metabolite_parent_fraction > 1)):
            raise PETKineticError("metabolite_parent_fraction must lie in [0,1]")
        scale = {"s": 1.0, "min": 1.0 / 60.0, "h": 1.0 / 3600.0}
        if time_unit not in scale:
            raise PETKineticError("time_unit must be 's', 'min', or 'h'")
        t = self.time_s * scale[time_unit]
        if np.any(np.diff(t) <= 0):
            raise PETKineticError("bloodproc time must be strictly increasing")
        return ArterialInputFunction(
            tuple(float(x) for x in t),
            tuple(float(x) for x in self.plasma_radioactivity),
            tuple(float(x) for x in self.metabolite_parent_fraction),
            float(delay),
        )


def read_bloodproc_tsv(tsv_path: str | Path, json_path: str | Path | None = None) -> BloodProcessedTable:
    path = Path(tsv_path)
    if not path.is_file():
        raise PETKineticError(f"bloodproc TSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise PETKineticError("bloodproc TSV has no header")
        fields = tuple(reader.fieldnames)
        required = {"time"}
        if not required.issubset(fields):
            raise PETKineticError("bloodproc TSV requires a 'time' column")
        rows = list(reader)
    if not rows:
        raise PETKineticError("bloodproc TSV is empty")

    def column(name: str) -> np.ndarray | None:
        if name not in fields:
            return None
        out = []
        for r in rows:
            value = r.get(name, "")
            if value is None or value == "":
                out.append(np.nan)
            else:
                try:
                    out.append(float(value))
                except ValueError as exc:
                    raise PETKineticError(f"non-numeric value in bloodproc column {name}") from exc
        return np.asarray(out, dtype=float)

    t = column("time")
    assert t is not None
    if np.any(~np.isfinite(t)) or np.any(t < 0) or np.any(np.diff(t) <= 0):
        raise PETKineticError("bloodproc time must be finite, non-negative and strictly increasing")
    plasma = column("plasma_radioactivity")
    parent = column("metabolite_parent_fraction")
    input_fn = column("input_function")
    if plasma is not None and np.any(~np.isfinite(plasma)):
        raise PETKineticError("plasma_radioactivity contains missing/non-finite values")
    if parent is not None and (np.any(~np.isfinite(parent)) or np.any((parent < 0) | (parent > 1))):
        raise PETKineticError("metabolite_parent_fraction must be finite in [0,1]")
    if input_fn is not None and np.any(~np.isfinite(input_fn)):
        raise PETKineticError("input_function contains missing/non-finite values")

    metadata: dict[str, Any] = {}
    if json_path is not None:
        jp = Path(json_path)
        if not jp.is_file():
            raise PETKineticError(f"bloodproc JSON not found: {jp}")
        metadata = json.loads(jp.read_text(encoding="utf-8"))

    return BloodProcessedTable(
        time_s=t,
        plasma_radioactivity=plasma,
        metabolite_parent_fraction=parent,
        input_function=input_fn,
        columns=fields,
        metadata=metadata,
    )

# BIDS 1.11.x uses recording-labelled *_blood.tsv / *_blood.json for raw blood
# recording. Keep the earlier `read_bloodproc_tsv` adapter as a compatibility
# layer for derivative/legacy tabular products, but expose the normative raw
# blood recording contract explicitly.
BIDS_BLOOD_CONTRACT_VERSION = "1.11.x-b19"


def read_bids_blood_tsv(tsv_path: str | Path, json_path: str | Path) -> BloodProcessedTable:
    """Read a normative BIDS PET blood-recording table and validate its sidecar flags.

    The current BIDS PET specification defines `*_blood.tsv` / sidecar JSON with
    required availability flags and a `time` column in seconds.  This adapter
    deliberately does not apply dispersion or metabolite correction itself;
    those are provenance-bearing processing steps that must be represented by
    their metadata and/or an explicit downstream derivative.
    """
    table = read_bloodproc_tsv(tsv_path, json_path)
    meta = table.metadata
    required_flags = ("PlasmaAvail", "MetaboliteAvail", "WholeBloodAvail", "DispersionCorrected")
    missing = [k for k in required_flags if k not in meta]
    if missing:
        raise PETKineticError(f"BIDS blood JSON missing required flags: {missing}")
    for key in required_flags:
        if not isinstance(meta[key], bool):
            raise PETKineticError(f"BIDS blood metadata {key} must be boolean")
    if meta["PlasmaAvail"] and table.plasma_radioactivity is None:
        raise PETKineticError("PlasmaAvail=true requires plasma_radioactivity")
    if meta["MetaboliteAvail"] and table.metabolite_parent_fraction is None:
        raise PETKineticError("MetaboliteAvail=true requires metabolite_parent_fraction")
    if meta["WholeBloodAvail"] and "whole_blood_radioactivity" not in table.columns:
        raise PETKineticError("WholeBloodAvail=true requires whole_blood_radioactivity")
    if meta["DispersionCorrected"] and "DispersionConstant" not in meta:
        raise PETKineticError(
            "DispersionCorrected=true requires an explicit dispersion metadata contract"
        )
    return table

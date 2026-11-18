#!/usr/bin/env python3
"""Donor-level pseudobulk pathway audit for public GEO GSE157783.

The dense UMI matrix is streamed gene-by-gene. Nuclei are never treated as
independent biological replicates. Outputs contain group-level statistics only.
This is a transcriptomic pathway proxy, not a measure of metabolic flux,
mitochondrial respiration, ATP production, or causal initiation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import re
import tarfile
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

DATASET = "GSE157783"
FILES = {
    "umi_matrix": "GSE157783_IPDCO_hg_midbrain_UMI.tar.gz",
    "cell_metadata": "GSE157783_IPDCO_hg_midbrain_cell.tar.gz",
    "gene_metadata": "GSE157783_IPDCO_hg_midbrain_genes.tar.gz",
}
MAPPING_COMMIT = "2d071afad9525a24dda6ae219928d1c3d5272f84"
MAPPING_URL = (
    "https://raw.githubusercontent.com/SpielmannLab/pd_human_midbrain_snrnaseq/"
    + MAPPING_COMMIT
    + "/scrnaseq/data/ensg_biomart_20191206.tsv"
)
REACTOME_GMT_URL = "https://reactome.org/download/current/ReactomePathways.gmt.zip"
REACTOME_VERSION_URL = "https://reactome.org/ContentService/data/database/version"

# These pathway stable identifiers were checked against the current Reactome
# Homo sapiens pathway catalogue (18 Nov 2026).
CORE_PATHWAYS = {
    "glycolysis": ("R-HSA-70171", "Glycolysis"),
    "pyruvate_metabolism": ("R-HSA-70268", "Pyruvate metabolism"),
    "protein_lipoylation": ("R-HSA-9857492", "Protein lipoylation"),
    "tca_cycle": ("R-HSA-71403", "Citric acid cycle (TCA cycle)"),
    "respiratory_electron_transport": ("R-HSA-611105", "Respiratory electron transport"),
    "pink1_prkn_mitophagy": ("R-HSA-5205685", "PINK1-PRKN Mediated Mitophagy"),
    "macroautophagy": ("R-HSA-1632852", "Macroautophagy"),
    "lysosome_vesicle_biogenesis": ("R-HSA-432720", "Lysosome Vesicle Biogenesis"),
}

MIN_CELLS_PER_DONOR_CELLTYPE = 20
MIN_DONORS_PER_GROUP = 3
MIN_GENES_PER_PATHWAY = 5
MARKER_GENES = (
    "PFKFB2", "PDHA1", "DLAT", "DLD", "PINK1", "PRKN", "LRRK2",
    "SNCA", "GBA1", "VPS35", "ATP13A2", "TMEM175", "RAB10",
    "LAMP1", "NDUFS1", "COX5A", "ATP5F1A",
)


def fetch_bytes(url: str, timeout: int = 180) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Neuro-Twin-public-research/0.1",
            "Accept-Encoding": "identity",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
    if not data:
        raise RuntimeError(f"Empty response from {url}")
    return data


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_tar_tsv(path: Path, suffix: str) -> io.TextIOWrapper:
    """Return a streaming text wrapper for the single matching TSV TAR member."""
    archive = tarfile.open(path, mode="r:gz")
    candidates = [
        member for member in archive.getmembers()
        if member.isfile() and member.name.lower().endswith(suffix.lower())
    ]
    if len(candidates) != 1:
        archive.close()
        raise RuntimeError(
            f"{path.name}: expected one {suffix} member, found {len(candidates)}"
        )
    stream = archive.extractfile(candidates[0])
    if stream is None:
        archive.close()
        raise RuntimeError(f"Cannot read member {candidates[0].name}")
    # The caller closes this wrapper; close the TAR archive when the stream closes.
    wrapper = io.TextIOWrapper(stream, encoding="utf-8", errors="strict", newline="")
    wrapper._neuro_twin_archive = archive  # type: ignore[attr-defined]
    return wrapper


def close_tar_tsv(stream: io.TextIOWrapper) -> None:
    archive = getattr(stream, "_neuro_twin_archive", None)
    stream.close()
    if archive is not None:
        archive.close()


def load_cells(path: Path) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    with read_tar_tsv(path, ".tsv") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"barcode", "cell_ontology", "patient"}
        if set(reader.fieldnames or ()) != required:
            raise RuntimeError(f"Unexpected cell metadata columns: {reader.fieldnames}")
        barcodes: list[str] = []
        cell_types: list[str] = []
        patients: list[str] = []
        seen: set[str] = set()
        for row in reader:
            barcode = row["barcode"].strip()
            cell_type = row["cell_ontology"].strip()
            patient = row["patient"].strip()
            if not barcode or not cell_type or not patient:
                raise RuntimeError("Blank barcode, cell_ontology, or patient in metadata")
            if barcode in seen:
                raise RuntimeError(f"Duplicate cell barcode in metadata: {barcode}")
            seen.add(barcode)
            barcodes.append(barcode)
            cell_types.append(cell_type)
            patients.append(patient)

    unique_patients = sorted(set(patients))
    if len(unique_patients) != 11:
        raise RuntimeError(f"Expected 11 donors from source study, found {unique_patients}")
    group_by_patient: dict[str, str] = {}
    for patient in unique_patients:
        upper = patient.upper()
        if upper.startswith("PD"):
            group_by_patient[patient] = "PD"
        elif upper.startswith("C"):
            group_by_patient[patient] = "control"
        else:
            raise RuntimeError(f"Unrecognised source donor label: {patient}")
    if sum(value == "PD" for value in group_by_patient.values()) != 5:
        raise RuntimeError(f"Expected 5 idiopathic PD donors, found {group_by_patient}")
    if sum(value == "control" for value in group_by_patient.values()) != 6:
        raise RuntimeError(f"Expected 6 control donors, found {group_by_patient}")
    return barcodes, cell_types, patients, group_by_patient


def load_gene_table(path: Path) -> list[dict[str, Any]]:
    with read_tar_tsv(path, ".tsv") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {
            "gene", "row", "vst.mean", "vst.variance",
            "vst.variance.expected", "vst.variance.standardized", "vst.variable",
        }
        if set(reader.fieldnames or ()) != required:
            raise RuntimeError(f"Unexpected gene metadata columns: {reader.fieldnames}")
        genes = []
        for item in reader:
            gene_id = item["gene"].strip()
            row_number = int(item["row"])
            if not gene_id or row_number < 1:
                raise RuntimeError(f"Invalid gene metadata row: {item}")
            genes.append({"gene_id": gene_id, "matrix_row": row_number})
    genes.sort(key=lambda item: item["matrix_row"])
    expected = list(range(1, len(genes) + 1))
    observed = [int(item["matrix_row"]) for item in genes]
    if observed != expected:
        raise RuntimeError("Gene metadata row numbers are not a contiguous 1-based matrix index")
    gene_ids = [item["gene_id"] for item in genes]
    if len(gene_ids) != len(set(gene_ids)):
        raise RuntimeError("Duplicate Ensembl gene IDs in gene metadata")
    return genes


def load_symbol_mapping() -> tuple[dict[str, str], str]:
    data = fetch_bytes(MAPPING_URL)
    mapping: dict[str, str] = {}
    protein_coding: set[str] = set()
    for row in csv.DictReader(io.StringIO(data.decode("utf-8")), delimiter="\t"):
        gene_id = row.get("Gene stable ID", "").strip()
        symbol = row.get("HGNC symbol", "").strip()
        gene_type = row.get("Gene type", "").strip()
        if not gene_id or not symbol:
            continue
        # Prefer protein-coding assignments if one stable ID has duplicate rows.
        if gene_type == "protein_coding":
            mapping[gene_id] = symbol
            protein_coding.add(gene_id)
        elif gene_id not in protein_coding:
            mapping.setdefault(gene_id, symbol)
    return mapping, sha256_bytes(data)


def load_reactome_sets() -> tuple[dict[str, set[str]], dict[str, Any]]:
    data = fetch_bytes(REACTOME_GMT_URL)
    version_raw = fetch_bytes(REACTOME_VERSION_URL).decode("utf-8").strip()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        gmt_files = [name for name in archive.namelist() if name.lower().endswith(".gmt")]
        if len(gmt_files) != 1:
            raise RuntimeError(f"Expected one GMT file in Reactome zip, found {gmt_files}")
        text = archive.read(gmt_files[0]).decode("utf-8")
    by_id: dict[str, set[str]] = {}
    labels: dict[str, str] = {}
    for line in text.splitlines():
        fields = line.rstrip("\r\n").split("\t")
        if len(fields) < 3:
            continue
        label = fields[0]
        match = re.search(r"R-HSA-\d+", label + " " + fields[1])
        if not match:
            continue
        stable_id = match.group(0)
        by_id[stable_id] = {gene.strip().upper() for gene in fields[2:] if gene.strip()}
        labels[stable_id] = label
    selected: dict[str, set[str]] = {}
    metadata: dict[str, Any] = {}
    for key, (stable_id, expected_label) in CORE_PATHWAYS.items():
        genes = by_id.get(stable_id)
        if not genes:
            # A conservative name fallback, still requiring a Homo sapiens stable ID.
            matches = [
                (candidate_id, candidate_genes)
                for candidate_id, candidate_genes in by_id.items()
                if expected_label.lower() in labels[candidate_id].lower()
            ]
            if len(matches) != 1:
                metadata[key] = {
                    "stable_id": stable_id,
                    "expected_label": expected_label,
                    "status": "not_found_or_ambiguous_in_gmt",
                }
                continue
            stable_id, genes = matches[0]
        selected[key] = genes
        metadata[key] = {
            "stable_id": stable_id,
            "label": labels.get(stable_id, expected_label),
            "source_gene_count": len(genes),
            "status": "loaded",
        }
    provenance = {
        "url": REACTOME_GMT_URL,
        "sha256": sha256_bytes(data),
        "reactome_content_service_version": version_raw,
        "gmt_member": gmt_files[0],
        "pathways": metadata,
    }
    return selected, provenance


def build_group_index(
    matrix_barcodes: list[str],
    cell_barcodes: list[str],
    cell_types: list[str],
    patients: list[str],
    group_by_patient: dict[str, str],
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, int]]:
    if len(matrix_barcodes) != len(set(matrix_barcodes)):
        raise RuntimeError("Duplicate cell barcode in matrix header")
    if set(matrix_barcodes) != set(cell_barcodes):
        missing = len(set(matrix_barcodes) - set(cell_barcodes))
        extra = len(set(cell_barcodes) - set(matrix_barcodes))
        raise RuntimeError(
            f"Matrix/cell metadata barcode mismatch: in matrix only={missing}, metadata only={extra}"
        )
    metadata_by_barcode = {
        barcode: (cell_type, patient)
        for barcode, cell_type, patient in zip(cell_barcodes, cell_types, patients)
    }
    ordered = [metadata_by_barcode[barcode] for barcode in matrix_barcodes]
    group_keys = sorted(set((patient, cell_type) for cell_type, patient in ordered))
    group_lookup = {key: i for i, key in enumerate(group_keys)}
    group_index = np.fromiter(
        (group_lookup[(patient, cell_type)] for cell_type, patient in ordered),
        dtype=np.int32,
        count=len(ordered),
    )
    group_records = [
        {
            "patient": patient,
            "diagnosis_group": group_by_patient[patient],
            "cell_type": cell_type,
        }
        for patient, cell_type in group_keys
    ]
    return group_index, group_records, group_lookup


def stream_pseudobulk(
    matrix_path: Path,
    expected_cells: int,
    genes: list[dict[str, Any]],
    group_index: np.ndarray,
    number_of_groups: int,
) -> tuple[np.ndarray, int]:
    """Stream dense genes x cells TSV into a compact genes x donor-celltype matrix."""
    counts = np.zeros((len(genes), number_of_groups), dtype=np.int64)
    with read_tar_tsv(matrix_path, ".tsv") as stream:
        header = stream.readline().rstrip("\r\n").split("\t")
        if len(header) != expected_cells:
            raise RuntimeError(
                f"Matrix header has {len(header)} columns; expected {expected_cells}"
            )
        for row_index, line in enumerate(stream):
            if row_index >= len(genes):
                raise RuntimeError("Matrix has more gene rows than gene metadata")
            values = np.fromstring(line.rstrip("\r\n"), sep="\t", dtype=np.int64)
            if values.size != expected_cells:
                raise RuntimeError(
                    f"Matrix row {row_index + 1} has {values.size} values; expected {expected_cells}"
                )
            if np.any(values < 0):
                raise RuntimeError(f"Negative UMI count in matrix row {row_index + 1}")
            aggregated = np.bincount(
                group_index, weights=values, minlength=number_of_groups
            )
            counts[row_index, :] = np.rint(aggregated).astype(np.int64)
        row_count = row_index + 1 if "row_index" in locals() else 0
    if row_count != len(genes):
        raise RuntimeError(
            f"Matrix has {row_count} gene rows but gene metadata has {len(genes)}"
        )
    return counts, row_count


def exact_permutation_p(scores: np.ndarray, pd_mask: np.ndarray) -> float:
    """Two-sided exact label permutation; donors, not nuclei, are permuted."""
    n = scores.size
    n_pd = int(pd_mask.sum())
    observed = float(scores[pd_mask].mean() - scores[~pd_mask].mean())
    extreme = 0
    total = 0
    tolerance = 1e-12
    for indices in itertools.combinations(range(n), n_pd):
        chosen = np.zeros(n, dtype=bool)
        chosen[list(indices)] = True
        difference = float(scores[chosen].mean() - scores[~chosen].mean())
        total += 1
        if abs(difference) >= abs(observed) - tolerance:
            extreme += 1
    return extreme / total if total else 1.0


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(np.asarray(p_values, dtype=float))
    result = np.ones(len(p_values), dtype=float)
    running = 1.0
    m = len(p_values)
    for rank_index in range(m - 1, -1, -1):
        original_index = int(order[rank_index])
        rank = rank_index + 1
        running = min(running, float(p_values[original_index]) * m / rank)
        result[original_index] = min(1.0, running)
    return result.tolist()


def summarize_tests(
    counts: np.ndarray,
    genes: list[dict[str, Any]],
    mapping: dict[str, str],
    pathway_sets: dict[str, set[str]],
    group_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    gene_index_by_symbol: dict[str, list[int]] = defaultdict(list)
    unmapped = 0
    for idx, gene in enumerate(genes):
        symbol = mapping.get(gene["gene_id"])
        if symbol:
            gene_index_by_symbol[symbol.upper()].append(idx)
        else:
            unmapped += 1

    cell_counts = np.asarray([0] * len(group_records), dtype=np.int64)
    # group_records already follows group index order; counts are separately
    # supplemented by cell-count data in the caller.
    library_sizes = counts.sum(axis=0, dtype=np.int64)
    log_cpm = np.log2(
        (counts.astype(np.float64) + 0.5)
        / (library_sizes.astype(np.float64)[None, :] + 1.0)
        * 1_000_000.0
    )

    cell_types = sorted({item["cell_type"] for item in group_records})
    tests: list[dict[str, Any]] = []
    marker_results: list[dict[str, Any]] = []
    for cell_type in cell_types:
        group_indices = [
            idx for idx, item in enumerate(group_records)
            if item["cell_type"] == cell_type
            and item["cell_count"] >= MIN_CELLS_PER_DONOR_CELLTYPE
            and library_sizes[idx] > 0
        ]
        if not group_indices:
            continue
        selected_records = [group_records[idx] for idx in group_indices]
        pd_mask = np.asarray(
            [item["diagnosis_group"] == "PD" for item in selected_records],
            dtype=bool,
        )
        n_pd = int(pd_mask.sum())
        n_control = int((~pd_mask).sum())
        if n_pd < MIN_DONORS_PER_GROUP or n_control < MIN_DONORS_PER_GROUP:
            for key, (stable_id, label) in CORE_PATHWAYS.items():
                tests.append({
                    "pathway_key": key,
                    "pathway_id": stable_id,
                    "cell_type": cell_type,
                    "status": "insufficient_donor_coverage",
                    "n_pd_donors": n_pd,
                    "n_control_donors": n_control,
                })
            continue

        donor_log_cpm = log_cpm[:, group_indices]
        gene_means = donor_log_cpm.mean(axis=1, keepdims=True)
        gene_sd = donor_log_cpm.std(axis=1, ddof=1, keepdims=True)
        variable = gene_sd[:, 0] > 1e-10
        z_scores = np.zeros_like(donor_log_cpm, dtype=np.float64)
        z_scores[variable, :] = (
            donor_log_cpm[variable, :] - gene_means[variable, :]
        ) / gene_sd[variable, :]

        for pathway_key, gene_symbols in pathway_sets.items():
            gene_indices = sorted({
                idx
                for symbol in gene_symbols
                for idx in gene_index_by_symbol.get(symbol, [])
                if variable[idx]
            })
            stable_id, label = CORE_PATHWAYS[pathway_key]
            if len(gene_indices) < MIN_GENES_PER_PATHWAY:
                tests.append({
                    "pathway_key": pathway_key,
                    "pathway_id": stable_id,
                    "pathway_label": label,
                    "cell_type": cell_type,
                    "status": "insufficient_gene_overlap",
                    "source_pathway_gene_count": len(gene_symbols),
                    "matrix_gene_overlap": len({
                        idx for symbol in gene_symbols
                        for idx in gene_index_by_symbol.get(symbol, [])
                    }),
                    "variable_gene_count": len(gene_indices),
                    "n_pd_donors": n_pd,
                    "n_control_donors": n_control,
                })
                continue
            donor_scores = z_scores[gene_indices, :].mean(axis=0)
            difference = float(donor_scores[pd_mask].mean() - donor_scores[~pd_mask].mean())
            pooled_var = (
                (n_pd - 1) * donor_scores[pd_mask].var(ddof=1)
                + (n_control - 1) * donor_scores[~pd_mask].var(ddof=1)
            ) / (n_pd + n_control - 2)
            pooled_sd = math.sqrt(max(0.0, float(pooled_var)))
            d = difference / pooled_sd if pooled_sd > 0 else 0.0
            correction = 1.0 - 3.0 / (4.0 * (n_pd + n_control) - 9.0)
            p_value = exact_permutation_p(donor_scores, pd_mask)
            tests.append({
                "pathway_key": pathway_key,
                "pathway_id": stable_id,
                "pathway_label": label,
                "cell_type": cell_type,
                "status": "tested",
                "n_pd_donors": n_pd,
                "n_control_donors": n_control,
                "mapped_genes_tested": len(gene_indices),
                "source_pathway_gene_count": len(gene_symbols),
                "mean_pd_score": float(donor_scores[pd_mask].mean()),
                "mean_control_score": float(donor_scores[~pd_mask].mean()),
                "mean_difference_pd_minus_control_z": difference,
                "hedges_g": correction * d,
                "exact_two_sided_permutation_p": p_value,
                "score_definition": "mean per-gene z-scored log2 CPM across donors within cell type",
            })

        for marker in MARKER_GENES:
            indexes = [
                idx for idx in gene_index_by_symbol.get(marker, [])
                if variable[idx]
            ]
            if not indexes:
                marker_results.append({
                    "gene_symbol": marker,
                    "cell_type": cell_type,
                    "status": "not_present_or_not_variable",
                    "n_pd_donors": n_pd,
                    "n_control_donors": n_control,
                })
                continue
            marker_scores = z_scores[indexes, :].mean(axis=0)
            marker_results.append({
                "gene_symbol": marker,
                "cell_type": cell_type,
                "status": "descriptive_only",
                "n_pd_donors": n_pd,
                "n_control_donors": n_control,
                "mean_pd_z": float(marker_scores[pd_mask].mean()),
                "mean_control_z": float(marker_scores[~pd_mask].mean()),
                "difference_pd_minus_control_z": float(
                    marker_scores[pd_mask].mean() - marker_scores[~pd_mask].mean()
                ),
                "interpretation": "exploratory transcript abundance; not causal evidence",
            })

    tested = [item for item in tests if item.get("status") == "tested"]
    q_values = benjamini_hochberg([
        float(item["exact_two_sided_permutation_p"]) for item in tested
    ])
    for item, q_value in zip(tested, q_values):
        item["benjamini_hochberg_q"] = q_value

    coverage = []
    for cell_type in cell_types:
        items = [
            item for item in group_records
            if item["cell_type"] == cell_type
            and item["cell_count"] >= MIN_CELLS_PER_DONOR_CELLTYPE
        ]
        coverage.append({
            "cell_type": cell_type,
            "donors_with_minimum_cells": len(items),
            "pd_donors": sum(item["diagnosis_group"] == "PD" for item in items),
            "control_donors": sum(item["diagnosis_group"] == "control" for item in items),
            "cells_total": sum(item["cell_count"] for item in items),
        })

    qc = {
        "genes_in_matrix": len(genes),
        "genes_with_symbol_mapping": len(genes) - unmapped,
        "genes_without_symbol_mapping": unmapped,
        "donor_celltype_pseudobulk_columns": len(group_records),
        "pathway_tests_attempted": len(tests),
        "pathway_tests_estimable": len(tested),
        "pathway_tests_fdr_q_below_0_05": sum(
            float(item.get("benjamini_hochberg_q", 1.0)) < 0.05 for item in tested
        ),
        "cell_type_coverage": coverage,
    }
    return tests, marker_results, qc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/tmp/neuro-twin-gse157783"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/gse157783_archive_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gse157783_pseudobulk_analysis.json"))
    args = parser.parse_args()

    archive_paths = {key: args.data_dir / filename for key, filename in FILES.items()}
    missing = [str(path) for path in archive_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing verified GEO archives: {missing}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cells = load_cells(archive_paths["cell_metadata"])
    genes = load_gene_table(archive_paths["gene_metadata"])
    mapping, mapping_hash = load_symbol_mapping()
    pathway_sets, reactome_provenance = load_reactome_sets()

    # Read matrix header separately, then validate the exact barcode mapping.
    with read_tar_tsv(archive_paths["umi_matrix"], ".tsv") as stream:
        matrix_barcodes = stream.readline().rstrip("\r\n").split("\t")
    cell_barcodes, cell_types, patients, group_by_patient = cells
    group_index, group_records, _ = build_group_index(
        matrix_barcodes, cell_barcodes, cell_types, patients, group_by_patient
    )

    # Count cells in each donor x cell-type pseudobulk column.
    group_cell_counts = np.bincount(group_index, minlength=len(group_records))
    for item, cell_count in zip(group_records, group_cell_counts):
        item["cell_count"] = int(cell_count)

    pseudobulk, matrix_rows = stream_pseudobulk(
        archive_paths["umi_matrix"],
        expected_cells=len(matrix_barcodes),
        genes=genes,
        group_index=group_index,
        number_of_groups=len(group_records),
    )
    tests, markers, qc = summarize_tests(
        pseudobulk, genes, mapping, pathway_sets, group_records
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "1.0.0",
        "dataset_id": DATASET,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "completed_donor_level_pseudobulk",
        "biological_design": {
            "donors_total": 11,
            "pd_donors": 5,
            "control_donors": 6,
            "cells_total": len(matrix_barcodes),
            "matrix_dimensions": {
                "genes": matrix_rows,
                "cells": len(matrix_barcodes),
                "matrix_layout": "genes_by_cells_dense_tsv",
            },
            "independent_unit": "donor",
            "pseudobulk_unit": "donor_by_cell_ontology",
            "minimum_cells_per_donor_celltype": MIN_CELLS_PER_DONOR_CELLTYPE,
        },
        "provenance": {
            "geo_archive_manifest": str(args.manifest),
            "geo_archives": manifest.get("archives", {}),
            "gene_symbol_mapping": {
                "source_url": MAPPING_URL,
                "source_commit": MAPPING_COMMIT,
                "sha256": mapping_hash,
                "annotation": "Ensembl gene stable ID to HGNC symbol; source author BioMart table dated 2019-12-06",
            },
            "reactome": reactome_provenance,
        },
        "analysis_specification": {
            "pathways": {
                key: {
                    "stable_id": CORE_PATHWAYS[key][0],
                    "expected_label": CORE_PATHWAYS[key][1],
                    "mapped_hgnc_gene_count": len(gene_set),
                }
                for key, gene_set in pathway_sets.items()
            },
            "normalization": "donor-celltype pseudobulk raw UMI counts -> log2 CPM with 0.5 pseudocount",
            "score": "within-celltype per-gene z-score across donors, averaged over pathway member genes",
            "test": "two-sided exact permutation over donor diagnosis labels",
            "multiple_testing": "Benjamini-Hochberg across all estimable pathway x cell-type tests",
            "effect_direction": "positive means higher transcriptomic module score in PD than controls",
            "limits": [
                "Postmortem cross-sectional data cannot establish temporal precedence or disease initiation.",
                "Transcript abundance is not ATP production, respiration, PDH enzyme activity, redox state, ROS, calcium flux, or mitochondrial membrane potential.",
                "This pilot is one small donor cohort (5 idiopathic PD, 6 controls); estimates are exploratory.",
                "Cell ontology categories are dataset annotations, not equivalent to purified cell populations.",
                "This analysis tests Parkinson midbrain only; Alzheimer disease routing requires a separately processed AD cohort.",
            ],
        },
        "quality_control": qc,
        "pathway_results": tests,
        "marker_results_descriptive_only": markers,
        "interpretation_boundary": (
            "Observational transcriptomic evidence only. No causal claim, treatment effect, "
            "or temporal ordering is inferred."
        ),
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"analysis_status={report['analysis_status']}")
    print(f"matrix={matrix_rows} genes x {len(matrix_barcodes)} cells")
    print(
        "donors=5 PD / 6 control; pathway tests="
        f"{qc['pathway_tests_estimable']}; q<0.05="
        f"{qc['pathway_tests_fdr_q_below_0_05']}"
    )
    for result in tests:
        if result.get("status") == "tested":
            print(
                f"{result['pathway_key']} | {result['cell_type']} | "
                f"PD-control={result['mean_difference_pd_minus_control_z']:.3f} "
                f"p={result['exact_two_sided_permutation_p']:.4f} "
                f"q={result.get('benjamini_hochberg_q', float('nan')):.4f}"
            )
    print(f"report={args.output}")


if __name__ == "__main__":
    main()

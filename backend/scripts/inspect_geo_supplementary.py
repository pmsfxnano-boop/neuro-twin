#!/usr/bin/env python3
"""Download and inventory public GSE157783 processed supplementary archives.

This is a source-integrity step only: it does not infer expression, cell types,
or disease effects. Archives are inspected in place and never extracted here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import tarfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FILES = {
    "umi_matrix": "GSE157783_IPDCO_hg_midbrain_UMI.tar.gz",
    "cell_metadata": "GSE157783_IPDCO_hg_midbrain_cell.tar.gz",
    "gene_metadata": "GSE157783_IPDCO_hg_midbrain_genes.tar.gz",
}
BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE157nnn/GSE157783/suppl/"


def download(url: str, destination: Path, attempts: int = 4) -> tuple[int, str, int | None]:
    """Stream a public GEO file to disk and return byte count and SHA-256."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        digest = hashlib.sha256()
        byte_count = 0
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Neuro-Twin-public-research/0.1 (GEO data audit)"},
        )
        try:
            request.add_header("Accept-Encoding", "identity")
            with urllib.request.urlopen(request, timeout=90) as response:
                content_length_header = response.headers.get("Content-Length")
                expected_bytes = int(content_length_header) if content_length_header else None
                with destination.open("wb") as output:
                    while True:
                        chunk = response.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        digest.update(chunk)
                        byte_count += len(chunk)
            if byte_count == 0:
                raise RuntimeError(f"empty response for {url}")
            if expected_bytes is not None and byte_count != expected_bytes:
                raise RuntimeError(
                    f"truncated response for {url}: received {byte_count}, expected {expected_bytes}"
                )
            return byte_count, digest.hexdigest(), expected_bytes
        except (OSError, urllib.error.URLError, RuntimeError) as exc:
            last_error = exc
            destination.unlink(missing_ok=True)
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {url}: {last_error}") from last_error


def preview_tsv(archive: tarfile.TarFile, member: tarfile.TarInfo) -> dict[str, Any]:
    """Read only a few initial TSV rows to establish schema without loading the matrix."""
    stream = archive.extractfile(member)
    if stream is None:
        raise RuntimeError(f"could not read TAR member: {member.name}")
    rows: list[dict[str, Any]] = []
    with stream, io.TextIOWrapper(stream, encoding="utf-8", errors="replace", newline="") as text:
        reader = csv.reader(text, delimiter=chr(9))
        for row in itertools.islice(reader, 5):
            rows.append({
                "field_count": len(row),
                "first_values": row[:12],
            })
    return {"preview_rows": rows}


def inspect_tar(path: Path) -> dict[str, Any]:
    """Read archive directory and TSV schema previews without extracting files."""
    try:
        with tarfile.open(path, mode="r:gz") as archive:
            members = archive.getmembers()
            regular = [member for member in members if member.isfile()]
            files = []
            for member in regular:
                record: dict[str, Any] = {
                    "name": member.name,
                    "size_bytes": member.size,
                    "suffixes": "".join(Path(member.name).suffixes),
                }
                if member.name.lower().endswith((".tsv", ".tsv.gz", ".txt")):
                    record.update(preview_tsv(archive, member))
                files.append(record)
    except (tarfile.TarError, OSError) as exc:
        raise RuntimeError(f"invalid gzip TAR archive: {path.name}: {exc}") from exc

    return {
        "member_count": len(members),
        "regular_file_count": len(regular),
        "directories": sorted({str(Path(m.name).parent) for m in members if m.isdir()}),
        "files": files,
    }


def build_manifest(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "dataset_id": "GSE157783",
        "source": "NCBI Gene Expression Omnibus",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "public_processed_archive_integrity_and_layout_audit",
        "interpretation": (
            "No expression analysis or biological inference is performed by this job."
        ),
        "archives": {},
    }

    for key, filename in FILES.items():
        url = BASE_URL + filename
        local_path = output_dir / filename
        last_error: Exception | None = None
        for validation_attempt in range(3):
            size_bytes, sha256, expected_bytes = download(url, local_path)
            try:
                archive_layout = inspect_tar(local_path)
                last_error = None
                break
            except (tarfile.TarError, EOFError, OSError, RuntimeError) as exc:
                last_error = exc
                local_path.unlink(missing_ok=True)
                if validation_attempt < 2:
                    time.sleep(2**validation_attempt)
        if last_error is not None:
            raise RuntimeError(f"could not validate GEO archive {filename}: {last_error}") from last_error
        manifest["archives"][key] = {
            "filename": filename,
            "url": url,
            "size_bytes": size_bytes,
            "content_length_bytes": expected_bytes,
            "sha256": sha256,
            **archive_layout,
        }
        # Keep raw data ephemeral on the runner; the artifact contains only metadata.
        local_path.unlink()

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/neuro-twin-gse157783"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/gse157783_archive_manifest.json"),
    )
    args = parser.parse_args()

    manifest = build_manifest(args.output_dir)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for key, archive in manifest["archives"].items():
        print(
            f"{key}: {archive['size_bytes']} bytes, "
            f"{archive['member_count']} members, sha256={archive['sha256']}"
        )
        for member in archive["files"][:20]:
            print(f"  {member['name']} ({member['size_bytes']} bytes)")
            for row in member.get("preview_rows", []):
                print(f"    TSV row: {row}")
    print(f"manifest: {args.manifest}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PATTERNS = [
    re.compile(r"(?i)-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"(?i)\bsk-[A-Za-z0-9_-]{20,}\b"),
]
FORBIDDEN_SUFFIXES = {".nii", ".nii.gz", ".dcm"}


def should_scan(path: Path) -> bool:
    if any(part in {".git", "node_modules", "dist", "__pycache__", ".venv"} for part in path.parts):
        return False
    return path.is_file() and path.stat().st_size <= 2_000_000


def main() -> int:
    violations: list[str] = []
    for path in ROOT.rglob("*"):
        if not should_scan(path):
            continue
        name = path.name.lower()
        if any(name.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES):
            violations.append(f"protected imaging object committed: {path.relative_to(ROOT)}")
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(text):
                violations.append(f"secret-like material detected: {path.relative_to(ROOT)} / {pattern.pattern}")

    if violations:
        print("Repository integrity FAILED:")
        print("\n".join(f"- {item}" for item in violations))
        return 1

    print("Repository integrity PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

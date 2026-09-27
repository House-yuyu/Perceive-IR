"""Check local model/source assets without loading PyTorch or downloading files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def check_assets(root: Path, manifest: dict, profile: str) -> list[dict]:
    if profile not in manifest["profiles"]:
        raise ValueError(f"unknown asset profile: {profile}")
    rows = []
    for asset in manifest["assets"]:
        if profile != "all" and profile not in asset["profiles"]:
            continue
        candidates = asset.get("any_of", [asset["path"]])
        present = any((root / name).is_file() and (root / name).stat().st_size > 0
                      for name in candidates)
        rows.append({"path": asset["path"], "status": "present" if present else "missing",
                     "purpose": asset["purpose"], "accepted_paths": candidates})
    return rows


def main() -> None:
    manifest = json.loads((PROJECT / "weight/manifest.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=manifest["profiles"], default="all")
    parser.add_argument("--root", type=Path, default=PROJECT)
    parser.add_argument("--json", action="store_true", help="print machine-readable results")
    args = parser.parse_args()
    rows = check_assets(args.root, manifest, args.profile)
    missing = sum(row["status"] == "missing" for row in rows)
    if args.json:
        print(json.dumps({"profile": args.profile, "missing": missing, "assets": rows}, indent=2))
    else:
        for row in rows:
            print(f"{row['status'].upper():7} {row['path']} ({row['purpose']})")
        print(f"\n{len(rows) - missing}/{len(rows)} assets present. See weight/README.md.")
        print("This checks nonempty files only; it does not verify model compatibility or datasets.")
    raise SystemExit(1 if missing else 0)


if __name__ == "__main__":
    main()

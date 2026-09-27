#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
ROOT_NAME = "perceiveIR"
DIRECTORIES = ("perceiveIR", "configs", "scripts", "tests")
FILES = ("README.md", "requirements.txt",
         ".gitignore", ".gitattributes", "weight/README.md", "weight/manifest.json",
         "data/README.md", "third_party/README.md")
FORBIDDEN_SUFFIXES = {".pth", ".pt", ".ckpt", ".pkl", ".safetensors", ".bin",
                      ".npy", ".npz", ".onnx", ".h5", ".hdf5"}


def source_files() -> list[tuple[Path, str]]:
    selected = []
    for directory in DIRECTORIES:
        for path in sorted((PROJECT / directory).rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(PROJECT)
            if ("__pycache__" in relative.parts or ".pytest_cache" in relative.parts
                    or path.suffix in {".pyc", ".orig"}):
                continue
            selected.append((path, f"{ROOT_NAME}/{relative.as_posix()}"))
    for name in FILES:
        path = PROJECT / name
        if not path.is_file():
            raise FileNotFoundError(path)
        selected.append((path, f"{ROOT_NAME}/{name}"))
    for name in ("LICENSE", "LICENSE.md", "NOTICE"):
        path = PROJECT / name
        if path.is_file():
            selected.append((path, f"{ROOT_NAME}/{name}"))
    for path, archive_name in selected:
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.stat().st_size > 20 * 2**20:
            raise RuntimeError(f"unsafe package member: {path}")
        if archive_name.startswith(f"{ROOT_NAME}/experiments/"):
            raise RuntimeError(f"experiment tree must not be copied: {archive_name}")
    archive_names = [name for _, name in selected]
    if len(archive_names) != len(set(archive_names)):
        raise RuntimeError("duplicate archive member name")
    return selected


def add_file(archive: tarfile.TarFile, path: Path, archive_name: str) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        data = handle.read()
    digest.update(data)
    info = tarfile.TarInfo(archive_name)
    info.size = len(data)
    info.mode = 0o755 if path.suffix == ".sh" else 0o644
    archive.addfile(info, io.BytesIO(data))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing archive: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    selected = source_files()
    fd, temporary = tempfile.mkstemp(prefix="perceiveIR_package_", suffix=".tar.gz.tmp",
                                    dir=destination.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "type": "source_and_documentation_no_weights_or_data",
                "files": {}}
    try:
        with tarfile.open(temporary_path, "w:gz") as archive:
            for path, archive_name in selected:
                manifest["files"][archive_name] = add_file(archive, path, archive_name)
            data = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            info = tarfile.TarInfo(f"{ROOT_NAME}/PACKAGE_MANIFEST.json")
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
        with tarfile.open(temporary_path, "r:gz") as archive:
            actual = {member.name for member in archive.getmembers()}
            expected = set(manifest["files"]) | {f"{ROOT_NAME}/PACKAGE_MANIFEST.json"}
            if actual != expected:
                raise RuntimeError("archive member list does not match manifest")
            for member in archive.getmembers():
                if member.name.endswith(tuple(FORBIDDEN_SUFFIXES)) or not member.isfile():
                    raise RuntimeError(f"weight or non-file member in archive: {member.name}")
                if member.name in manifest["files"]:
                    content = archive.extractfile(member)
                    assert content is not None
                    if hashlib.sha256(content.read()).hexdigest() != manifest["files"][member.name]:
                        raise RuntimeError(f"archive member hash mismatch: {member.name}")
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    print(json.dumps({"archive": str(destination), "file_count": len(manifest["files"]),
                      "bytes": destination.stat().st_size,
                      "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()

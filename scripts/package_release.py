import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    parser = argparse.ArgumentParser(description="Package the portable checkout without local training runs or caches")
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-weights", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = Path(args.output).resolve()
    if root == destination or root in destination.parents:
        raise ValueError("Write the ZIP outside the repository to avoid packaging itself.")
    ignored = {"runs", ".venv", "venv", "__pycache__", ".pytest_cache", ".git", "build", "dist"}
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in ignored or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.suffix in (".pyc", ".tmp") or (path.suffix == ".pt" and not args.include_weights):
            continue
        if relative.as_posix() == "RELEASE_MANIFEST.json":
            continue
        records.append({"file": relative.as_posix(), "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = root / "RELEASE_MANIFEST.json"
    manifest.write_text(json.dumps({"files": records, "includes_weights": args.include_weights}, indent=2), encoding="utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for record in records:
            archive.write(root / record["file"], f"STV-GLGFormer/{record['file']}")
        archive.write(manifest, "STV-GLGFormer/RELEASE_MANIFEST.json")
    print(json.dumps({"zip": str(destination), "bytes": destination.stat().st_size,
                      "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "files": len(records) + 1}, indent=2))


if __name__ == "__main__":
    main()

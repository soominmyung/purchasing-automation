"""
Keep the pipeline's history store in step with the company's shared history folder.

Meant to run on a machine that can see the network drive (the always-on SAP server), on a
schedule, e.g. nightly with Windows Task Scheduler. It walks every subfolder, compares each
PDF's fingerprint with what the store holds, and sends only what changed:

  new file      -> added
  changed file  -> its earlier version replaced
  deleted file  -> removed

How the folder is organised (by year, by supplier, ...) does not matter; each document's
header decides what it is about. The folder path is kept as the document's name, shown in
the report's "Referenced Documents", and a header that disagrees with a supplier folder is
reported as a warning.

Deletions are held back when they would remove more than half of the documents on file,
which is what a disconnected or empty drive looks like; pass --allow-mass-delete when that
is really intended.

  python scripts/sync_history_folder.py --root "Z:/Purchasing/History" --api https://<service>/api/ingest
      (API key from the PURCHASING_API_KEY environment variable)
  python scripts/sync_history_folder.py --root <folder> --local      (write to this machine's store)
  add --dry-run to list what would change without changing anything
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

MASS_DELETE_SHARE = 0.5


def scan(root: Path) -> dict[str, tuple[Path, str]]:
    """Every PDF under root: path inside the folder (with / separators) -> (file, fingerprint)."""
    found = {}
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix.lower() == ".pdf" and not f.name.startswith("~$"):
            found[f.relative_to(root).as_posix()] = (f, hashlib.sha256(f.read_bytes()).hexdigest())
    return found


class LocalStore:
    """Writes straight to this machine's store (for testing, or when the API runs here)."""

    def __init__(self):
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from services import history_folder
        from utils.pdf_utils import extract_text_from_pdf
        self.hf, self.extract = history_folder, extract_text_from_pdf

    def held(self) -> dict[str, dict]:
        return self.hf.folder_documents()

    def put(self, path: str, file: Path, fp: str) -> dict:
        try:
            return self.hf.store_folder_document(self.extract(file), path, fp)
        except ValueError as e:
            return {"path": path, "ok": False, "error": str(e)}

    def delete(self, path: str) -> dict:
        return {"ok": True, "chunks_removed": self.hf.remove_folder_document(path)}


class ApiStore:
    """Talks to the deployed pipeline's ingest API."""

    def __init__(self, base: str, key: str | None):
        import requests
        self.http, self.base = requests.Session(), base.rstrip("/")
        if key:
            self.http.headers["X-API-Key"] = key

    def held(self) -> dict[str, dict]:
        r = self.http.get(f"{self.base}/history/folder", timeout=60)
        r.raise_for_status()
        return r.json()["documents"]

    def put(self, path: str, file: Path, fp: str) -> dict:
        with open(file, "rb") as fh:
            r = self.http.put(f"{self.base}/history/folder", data={"path": path},
                              files={"file": (file.name, fh, "application/pdf")}, timeout=300)
        r.raise_for_status()
        return r.json()

    def delete(self, path: str) -> dict:
        r = self.http.delete(f"{self.base}/history/folder", params={"path": path}, timeout=60)
        r.raise_for_status()
        return r.json()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="top of the shared history folder")
    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument("--api", help="base URL of the ingest API, e.g. https://<service>/api/ingest")
    target.add_argument("--local", action="store_true", help="write to this machine's store")
    ap.add_argument("--dry-run", action="store_true", help="list changes without making them")
    ap.add_argument("--allow-mass-delete", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"Folder not found or not reachable: {root}")
    store = LocalStore() if args.local else ApiStore(args.api, os.environ.get("PURCHASING_API_KEY"))

    local, held = scan(root), store.held()
    to_add = [p for p in local if p not in held]
    to_replace = [p for p in local if p in held and held[p].get("fingerprint") != local[p][1]]
    to_delete = [p for p in held if p not in local]
    print(f"{datetime.now():%Y-%m-%d %H:%M}  {root}: {len(local)} files, {len(held)} held  ->  "
          f"{len(to_add)} new, {len(to_replace)} changed, {len(to_delete)} deleted")

    if held and len(to_delete) > MASS_DELETE_SHARE * len(held) and not args.allow_mass_delete:
        print(f"  Holding back {len(to_delete)} deletions (over {MASS_DELETE_SHARE:.0%} of documents on file). "
              "Check the drive is connected; rerun with --allow-mass-delete if this is intended.")
        to_delete = []

    if args.dry_run:
        for label, paths in (("new", to_add), ("changed", to_replace), ("deleted", to_delete)):
            for p in paths:
                print(f"  [{label}] {p}")
        return

    failures, warnings = [], []
    for p in to_add + to_replace:
        result = store.put(p, *local[p])
        if not result.get("ok"):
            failures.append(f"{p}: {result.get('error')}")
        elif result.get("warning"):
            warnings.append(f"{p}: {result['warning']}")
        else:
            about = result.get("supplier") or "-"
            print(f"  {'replaced' if p in to_replace else 'added':<8} {p}  (supplier {about}, "
                  f"items {', '.join(result.get('items') or result.get('item_codes') or []) or '-'})")
    for p in to_delete:
        store.delete(p)
        print(f"  removed  {p}")
    for w in warnings:
        print(f"  WARNING  {w}")
    for f_ in failures:
        print(f"  FAILED   {f_}")
    print(json.dumps({"added": len(to_add), "replaced": len(to_replace), "removed": len(to_delete),
                      "warnings": len(warnings), "failed": len(failures)}))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

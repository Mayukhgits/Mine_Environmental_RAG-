"""End-to-end check against a RUNNING server.

    python scripts/smoke_test.py path/to/file.pdf "Your question"
    python scripts/smoke_test.py --base http://localhost:8000 file.pdf "Your question"
"""
import argparse
import json
import sys

import requests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("question")
    ap.add_argument("--base", default="http://localhost:8000")
    a = ap.parse_args()
    api = a.base.rstrip("/") + "/api"

    h = requests.get(f"{api}/health", timeout=30)
    print("health:", h.status_code, h.json())

    with open(a.pdf, "rb") as f:
        up = requests.post(f"{api}/upload", files={"file": (a.pdf.replace("\\", "/").split("/")[-1], f, "application/pdf")}, timeout=900)
    print("upload:", up.status_code, up.json())
    up.raise_for_status()

    res = requests.post(f"{api}/chat", json={"message": a.question, "documents": [up.json()["doc_hash"]]}, timeout=300)
    print("chat:", res.status_code)
    print(json.dumps(res.json().get("response"), indent=2, ensure_ascii=False))
    print("meta:", res.json().get("meta"))
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Run a photo through a running Fridge Sloth server and print what it found.

Usage: python scripts/try_photo.py examples/photos/usda-0017.jpg [--scan-only]
"""

import base64
import json
import sys
import urllib.request
from typing import Any

BASE_URL = "http://localhost:8000"


def post(path: str, payload: dict[str, Any]) -> Any:
    req = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def main(photo: str, scan_only: bool) -> None:
    with open(photo, "rb") as f:
        image = base64.b64encode(f.read()).decode()

    scan = post("/api/scan", {"image": image})
    print(f"SCAN ({scan['seconds']}s, {scan['model']})")
    print("  seen:", ", ".join(f"{s['item']} [{s['amount']}]" for s in scan["seen"]))
    print("  staples:", ", ".join(f"{s['name']}={s['status']}" for s in scan["staples"]))
    print("  grocery list:", scan["grocery_list"])
    if scan_only:
        return

    recipes = post("/api/recipes", {"items": [s["item"] for s in scan["seen"]]})
    print(f"RECIPES ({recipes['seconds']}s)")
    for r in recipes["recipes"]:
        print(f"  - {r['name']} ({r['minutes']} min): {r['why_healthy']}")
        print(f"      uses: {r['uses']}  buy: {r['need_to_buy']}")
        for step in r["steps"]:
            print(f"      * {step}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], "--scan-only" in sys.argv)

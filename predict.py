"""Use the saved centers without fitting again: python predict.py --input examples/new_user.json."""
import argparse
import json
from pathlib import Path

from carecompass.decision import load_artifact, make_decision

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSON file containing the five user inputs")
    parser.add_argument("--model", type=Path, default=ROOT / "outputs/model.json")
    args = parser.parse_args()
    try:
        result = make_decision(json.loads(args.input.read_text(encoding="utf-8")), load_artifact(args.model))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Cannot assign a profile: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))

import csv
import json
import sys
from pathlib import Path


def csv_to_json(csv_path: str) -> None:
    csv_path = Path(csv_path)
    #Get the directory name and set the json_path to a file called metrics.json in that directory
    json_path = csv_path.parent / "metrics.json"

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        values = next(reader)

    result = {}
    for header, value in zip(headers, values):
        result[header] = value

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Written to {json_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_bfcl_scores_to_beaker_metrics.py <path_to_csv>")
        sys.exit(1)

    csv_to_json(sys.argv[1])
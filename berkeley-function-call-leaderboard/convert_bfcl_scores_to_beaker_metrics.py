import argparse
import csv
import json
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

def scores_to_json(dir_path: str) -> None:
    dir_path = Path(dir_path)

    # Find the only subdirectory (named after the model with "/" replaced by "_")
    subdirs = [p for p in dir_path.iterdir() if p.is_dir()]
    if len(subdirs) != 1:
        raise ValueError(f"Expected exactly 1 subdirectory in {dir_path}, found {len(subdirs)}")

    model_dir = subdirs[0]
    model_name_unescaped = model_dir.name
    print(f"Processing model: {model_name_unescaped} in directory: {model_dir}")

    # Find all BFCL_v3_<metric_name>_score.json files
    result = {}
    for score_file in sorted(model_dir.glob("BFCL_v3_*_score.json")):
        # Extract metric name from filename: BFCL_v3_<metric_name>_score.json
        metric_name = score_file.stem[len("BFCL_v3_"):-len("_score")]

        print(f"Loading score for metric: {metric_name} from file: {score_file}")
        with open(score_file) as f:
            #load the json from the first line of the file
            data = json.loads(f.readline())

        result[metric_name] = data["accuracy"]

    json_path = dir_path / f"metrics.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Written to {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV or BFCL score files to JSON.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--overall_csv", metavar="PATH", help="Path to a CSV file to convert to JSON.")
    group.add_argument("--ind_scores", metavar="DIR", help="Path to a directory containing a model subdirectory with BFCL score files.")

    args = parser.parse_args()

    if args.overall_csv:
        csv_to_json(args.overall_csv)
    elif args.ind_scores:
        scores_to_json(args.ind_scores)
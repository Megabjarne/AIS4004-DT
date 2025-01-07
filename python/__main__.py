import argparse
import pandas
from pathlib import Path


def load_data(path: Path) -> pandas.DataFrame:
    return pandas.read_csv(path)


def parse_arguments() -> argparse.NameSpace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default="data.csv", help="path to the csv file with data"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    data = load_data(args.data)
    print(data)


if __name__ == "__main__":
    main()

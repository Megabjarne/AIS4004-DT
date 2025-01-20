import argparse
import pandas
from pathlib import Path
from scipy import stats
import numpy as np
import grafana


def load_data(path: Path) -> pandas.DataFrame:
    data = pandas.read_csv(path)
    data.columns = list(s.strip() for s in data.columns)
    return data


def prepare_data(data: pandas.DataFrame) -> pandas.DataFrame:
    # remove rows with null values
    data = data[data.notnull().all(axis=1)]

    # remove consecutive duplicates
    not_duplicate = data.diff(-1).any(axis=1)
    not_duplicate[not_duplicate.size - 1] = True
    data = data[not_duplicate.values]

    # remove outliers
    non_static_columns = data.diff(-1).any(axis=0)
    data = data[
        (np.abs(stats.zscore(data[data.columns[non_static_columns]])) < 3).all(axis=1)
    ]

    return data


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default="data.csv", help="path to the csv file with data"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    data = load_data(args.data)
    print(f"loaded {len(data)} rows of data")
    clean_data = prepare_data(data)
    print(f"removed {len(data) - len(clean_data)} rows of data")
    print(data)

    grafana.push_data_sync(data)


if __name__ == "__main__":
    main()

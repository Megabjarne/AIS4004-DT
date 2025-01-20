import argparse
import pandas
from typing import Callable, Any
from pathlib import Path
from scipy import stats
import numpy as np
from itertools import combinations
import multiprocessing

import grafana
import model


def load_data(path: Path) -> pandas.DataFrame:
    data = pandas.read_csv(path)
    data.columns = list(s.strip() for s in data.columns)
    del data["index"]
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
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("upload", help="upload data to local postgres database")

    train_parser = subparsers.add_parser("train", help="train/fit a model on the data")
    train_parser.add_argument(
        "--model",
        type=str,
        default="random-forest",
        choices=["random-forest", "svc"],
        help="what model to use",
    )
    train_parser.add_argument(
        "--ncols", type=int, default=None, help="iteratively fit to subset of columns"
    )
    train_parser.add_argument(
        "--cols", type=str, default=None, nargs="+", help="columns to use for training"
    )

    return parser.parse_args()


def upload_data(data: pandas.DataFrame) -> None:
    grafana.push_data_sync(data)


def get_model_trainer(model_name: str) -> Callable:
    if model_name == "svc":
        model_trainer = model.train_svc
    elif model_name == "random-forest":
        model_trainer = model.train_randomforest
    else:
        raise ValueError("unknown model name")
    return model_trainer


def _model_score(model_name: str, data: pandas.DataFrame) -> tuple[float, float]:
    model_trainer = get_model_trainer(model_name)
    ((_, compressor_score), (_, turbine_score)) = model_trainer(data)
    return compressor_score, turbine_score


def train_model(
    data: pandas.DataFrame, model_name: str, ncols: int | None, cols: list[str] | None
) -> None:
    model_trainer: Callable[
        [pandas.DataFrame], tuple[tuple[Any, float], tuple[Any, float]]
    ]
    model_trainer = get_model_trainer(model_name)

    if ncols is None and cols is None:
        # we train on all the data
        model_trainer(data)
    elif ncols is None and cols is not None:
        # we train on the specified subset of columns
        # we check that the columns exist
        for col in cols:
            if col not in data.columns:
                raise ValueError(f"specified column {col} is not in the dataset")
        # we ensure the decay columns are included
        colset = set(cols)
        colset.add("GT Compressor decay state coefficient")
        colset.add("GT Turbine decay state coefficient")
        # we extract the subset of the data, and train
        subdata = data[list(colset)]
        model_trainer(subdata)
    elif ncols is not None and cols is None:
        # we pick out n columns and fit on that, we do this for all permutations of n columns
        data_columns = set(data.columns)
        data_columns.remove("GT Compressor decay state coefficient")
        data_columns.remove("GT Turbine decay state coefficient")

        if len(data_columns) < ncols:
            raise ValueError(
                f"cannot use more columns than there are in the data ({ncols} > {len(data_columns)})"
            )

        with open("permutation_results.csv", "w+") as fd:
            # write header
            fd.write(
                f"{','.join(f'column_{i}' for i in range(ncols))},compressor_score,turbine_score\n"
            )
            subsets = []
            subcols = []
            for subset in combinations(data_columns, ncols):
                # we reintroduce the decay columns
                colset = set(subset)
                colset.add("GT Compressor decay state coefficient")
                colset.add("GT Turbine decay state coefficient")
                # we extract the subset of the data, and train
                subdata = data[list(colset)]
                subsets.append(subdata)
                subcols.append(subset)
                # print(f"### Training on columns {colset}")
                # ((_, compressor_score), (_, turbine_score)) = model_trainer(subdata)
            with multiprocessing.Pool() as pool:
                for columns, (compressor_score, turbine_score) in zip(
                    subcols,
                    pool.starmap(
                        _model_score, [(model_name, data) for data in subsets]
                    ),
                ):
                    fd.write(
                        f"{','.join(columns)},{compressor_score},{turbine_score}\n"
                    )

    else:
        raise ValueError("ncols and cols are mutually exclusive")


def main() -> None:
    args = parse_arguments()

    data = load_data(args.data)
    print(f"loaded {len(data)} rows of data")
    clean_data = prepare_data(data)
    print(f"removed {len(data) - len(clean_data)} rows of data")

    if args.command == "upload":
        upload_data(data)
    elif args.command == "train":
        train_model(data, args.model, args.ncols, args.cols)


if __name__ == "__main__":
    main()

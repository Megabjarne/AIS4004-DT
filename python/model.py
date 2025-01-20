from sklearn import svm
from sklearn import ensemble
import pandas
import xgboost as xgb

DECAY_CUTOFF = 0.99


def split_dataset(
    data: pandas.DataFrame,
) -> tuple[
    pandas.DataFrame,
    pandas.DataFrame,
    pandas.DataFrame,
    pandas.DataFrame,
    pandas.DataFrame,
    pandas.DataFrame,
]:
    data = data.copy().sample(frac=1)

    compressor_decay = data["GT Compressor decay state coefficient"] < DECAY_CUTOFF
    turbine_decay = data["GT Turbine decay state coefficient"] < DECAY_CUTOFF
    del data["GT Compressor decay state coefficient"]
    del data["GT Turbine decay state coefficient"]
    del data["index"]

    training_samples = int(len(data) * 0.7)

    training_data = data[:training_samples]
    evaluation_data = data[training_samples:]

    training_compressor_decay = compressor_decay[:training_samples]
    evaluation_compressor_decay = compressor_decay[training_samples:]

    training_turbine_decay = turbine_decay[:training_samples]
    evaluation_turbine_decay = turbine_decay[training_samples:]

    return (
        training_data,
        evaluation_data,
        training_compressor_decay,
        evaluation_compressor_decay,
        training_turbine_decay,
        evaluation_turbine_decay,
    )


def train_svm(
    data: pandas.DataFrame,
) -> tuple[tuple[svm.SVC, float], tuple[svm.SVC, float]]:
    (
        training_data,
        evaluation_data,
        training_compressor_decay,
        evaluation_compressor_decay,
        training_turbine_decay,
        evaluation_turbine_decay,
    ) = split_dataset(data)

    # Train models
    compressor_model = svm.SVC()
    turbine_model = svm.SVC()

    compressor_model.fit(training_data, training_compressor_decay)
    turbine_model.fit(training_data, training_turbine_decay)

    # Evaluate models
    predicted_compressor = compressor_model.predict(evaluation_data)
    predicted_turbine = turbine_model.predict(evaluation_data)

    compressor_correct = (predicted_compressor == evaluation_compressor_decay).sum()
    turbine_correct = (predicted_turbine == evaluation_turbine_decay).sum()

    compressor_score = compressor_correct / len(evaluation_compressor_decay)
    turbine_score = turbine_correct / len(evaluation_turbine_decay)

    print(f"SVM score {compressor_score=}, {turbine_score=}")
    return ((compressor_model, compressor_score), (turbine_model, turbine_score))


def train_randomforest(
    data: pandas.DataFrame,
) -> tuple[
    tuple[ensemble.RandomForestClassifier, float],
    tuple[ensemble.RandomForestClassifier, float],
]:
    (
        training_data,
        evaluation_data,
        training_compressor_decay,
        evaluation_compressor_decay,
        training_turbine_decay,
        evaluation_turbine_decay,
    ) = split_dataset(data)

    # Train models
    compressor_model = ensemble.RandomForestClassifier()
    turbine_model = ensemble.RandomForestClassifier()

    compressor_model.fit(training_data, training_compressor_decay)
    turbine_model.fit(training_data, training_turbine_decay)

    # Evaluate models
    predicted_compressor = compressor_model.predict(evaluation_data)
    predicted_turbine = turbine_model.predict(evaluation_data)

    compressor_correct = (predicted_compressor == evaluation_compressor_decay).sum()
    turbine_correct = (predicted_turbine == evaluation_turbine_decay).sum()

    compressor_score = compressor_correct / len(evaluation_compressor_decay)
    turbine_score = turbine_correct / len(evaluation_turbine_decay)

    print(f"RandomForest score {compressor_score=}, {turbine_score=}")
    return ((compressor_model, compressor_score), (turbine_model, turbine_score))

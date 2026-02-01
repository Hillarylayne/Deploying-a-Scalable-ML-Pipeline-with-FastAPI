import numpy as np
import pandas as pd

from ml.data import apply_label, process_data
from ml.model import compute_model_metrics, inference, train_model

# Keep consistent with train_model.py
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def test_apply_label():
    """apply_label should map binary output to the expected string labels."""
    assert apply_label([1]) == ">50K"
    assert apply_label([0]) == "<=50K"


def test_compute_model_metrics_returns_expected_values():
    """compute_model_metrics should return correct precision/recall/F1 on a known example."""
    y = np.array([1, 0, 1, 0])
    preds = np.array([1, 0, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    # Manually computed: TP=1, FP=0, FN=1
    assert precision == 1.0
    assert recall == 0.5
    assert abs(fbeta - (2 * 1.0 * 0.5 / (1.0 + 0.5))) < 1e-8


def test_train_and_inference_output_shape():
    """Model should train and inference should return predictions of the correct length."""
    df = pd.read_csv("data/census.csv").head(300).copy()

    X, y, encoder, lb = process_data(
        df,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )

    model = train_model(X, y)
    preds = inference(model, X)

    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X.shape[0]


from pathlib import Path

import pandas as pd

SAMPLE = Path(__file__).resolve().parents[2] / "frontend" / "public" / "sample_campaign.csv"
REQUIRED = ["customer_id", "discount_given", "purchased"]  # same rule the upload page enforces


def load():
    return pd.read_csv(SAMPLE)


def test_sample_has_required_columns():
    df = load()
    for col in REQUIRED:
        assert col in df.columns


def test_sample_has_customer_features():
    # The upload page asks for at least one feature column besides the three required ones
    assert len(load().columns) >= 4


def test_binary_columns_only_contain_0_and_1():
    df = load()
    assert set(df["discount_given"].unique()) <= {0, 1}
    assert set(df["purchased"].unique()) <= {0, 1}


def test_sample_does_not_leak_ground_truth():
    assert "true_ite" not in load().columns


def test_sample_has_no_duplicate_customers():
    assert load()["customer_id"].is_unique

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "data"))
from generate_data import generate  # noqa: E402


def test_shape_and_columns():
    df = generate(n=500, seed=1)
    assert len(df) == 500
    for col in ["customer_id", "discount_given", "purchased", "true_ite"]:
        assert col in df.columns


def test_binary_columns():
    df = generate(n=500, seed=1)
    assert set(df["discount_given"].unique()) <= {0, 1}
    assert set(df["purchased"].unique()) <= {0, 1}


def test_confounding_exists():
    # Treated customers should have higher past spend on average (selection bias)
    df = generate(n=5000, seed=1)
    means = df.groupby("discount_given")["past_spend"].mean()
    assert means[1] > means[0]

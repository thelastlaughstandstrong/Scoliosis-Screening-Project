import pandas as pd
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INPUT  = _PROJECT_ROOT / "Data" / "Processed" / "aggregated_trials.csv"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "Data" / "Processed" / "screened_trials.csv"

# Threshold from Adam's forward bend screening literature (scoliometer ATR >= 5 deg).
# Applied here to a camera-derived proxy angle — NOT a clinical ATR measurement.
# Output column is a rule-based screening flag, NOT a ground-truth label.
# Do not use proxy_screening_flag as a training target without an independent
# clinical reference (scoliometer, X-ray Cobb angle, or clinician assessment).
ATR_PROXY_THRESHOLD_DEG = 5.0


def assign_screening_flags(df: pd.DataFrame) -> pd.DataFrame:
    if "atr_proxy_p95_deg" in df.columns:
        atr_col = "atr_proxy_p95_deg"
    elif "atr_proxy_deg_max" in df.columns:
        atr_col = "atr_proxy_deg_max"
    else:
        raise ValueError(
            "Dataset must contain 'atr_proxy_p95_deg' or 'atr_proxy_deg_max'."
        )

    df = df.copy()
    df["proxy_screening_flag"] = (df[atr_col] >= ATR_PROXY_THRESHOLD_DEG).astype(int)
    return df


if __name__ == "__main__":
    if len(sys.argv) == 1:
        input_csv = _DEFAULT_INPUT
        output_csv = _DEFAULT_OUTPUT
    elif len(sys.argv) == 3:
        input_csv = Path(sys.argv[1])
        output_csv = Path(sys.argv[2])
    else:
        print("Usage: python add_labels.py [<input_csv> <output_csv>]")
        sys.exit(1)

    df = pd.read_csv(input_csv)
    df = assign_screening_flags(df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[DONE] Screened dataset saved to: {output_csv}")
    print(f"       Screening flag positive (proxy >= {ATR_PROXY_THRESHOLD_DEG} deg): "
          f"{df['proxy_screening_flag'].sum()} / {len(df)}")
    print(f"       NOTE: proxy_screening_flag is rule-based, not a clinical ground truth.")
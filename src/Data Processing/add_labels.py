import pandas as pd
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INPUT  = _PROJECT_ROOT / "Data" / "Processed" / "aggregated_trials.csv"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "Data" / "Processed" / "labeled_trials.csv"

ATR_THRESHOLD_DEG = 5.0


def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    # Support output from aggregate_all_trials.py (atr_proxy_p95_deg)
    # and legacy aggregate_all_subjects.py (atr_proxy_deg_max)
    if "atr_proxy_p95_deg" in df.columns:
        atr_col = "atr_proxy_p95_deg"
    elif "atr_proxy_deg_max" in df.columns:
        atr_col = "atr_proxy_deg_max"
    else:
        raise ValueError(
            "Dataset must contain 'atr_proxy_p95_deg' or 'atr_proxy_deg_max'."
        )

    df = df.copy()
    df["risk_label"] = (df[atr_col] >= ATR_THRESHOLD_DEG).astype(int)
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
    df = assign_labels(df)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[DONE] Labeled dataset saved to: {output_csv}")
    print(f"       Positive (risk_label=1): {df['risk_label'].sum()} / {len(df)}")

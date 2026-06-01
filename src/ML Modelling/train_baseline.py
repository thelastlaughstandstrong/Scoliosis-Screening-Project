import sys
import joblib
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

# Features from aggregate_all_trials.py output
FEATURE_COLS = [
    "standing_symmetry_median",
    "standing_midline_offset_p95_px",
    "trunk_curvature_p95_deg",
    "atr_proxy_p95_deg",
    "rib_hump_proxy_p95_px",
    "trunk_inclination_2d_median_deg",
    "trunk_inclination_2d_p95_deg",
    "pose_coverage_ratio",
]
TARGET_COL = "risk_label"


def load_dataset(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    available = [c for c in FEATURE_COLS if df[c].notna().any()]
    X = df[available].fillna(0)
    y = df[TARGET_COL]
    return X, y


def train(csv_path: Path, model_out: Path) -> None:
    X, y = load_dataset(csv_path)

    if len(y.unique()) < 2:
        print("[WARN] Only one class present — cannot train a meaningful classifier.")
        print("       Collect data with both positive and negative labels first.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"Cross-val ROC-AUC: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print("\nTest set classification report:")
    print(classification_report(y_test, y_pred))

    if len(y_test.unique()) > 1:
        auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        print(f"Test ROC-AUC: {auc:.3f}")

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_out)
    print(f"\n[DONE] Model saved to: {model_out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_baseline.py <labeled_csv> <model_output_pkl>")
        sys.exit(1)

    train(Path(sys.argv[1]), Path(sys.argv[2]))

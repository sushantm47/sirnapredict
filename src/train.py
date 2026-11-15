"""
Train the knockdown-efficacy model.

The important choice here is how the data is split. siRNAs that target the same
gene share sequence context, so a random split lets the model see near-duplicates
of its test data and look far better than it is. Every split below is grouped by
target gene, so the model is always scored on genes it has never seen - which is
the situation it faces when someone designs siRNAs for a new target.

Three models are compared on the same splits:

  rules   - the Reynolds et al. (2004) checklist alone, the pre-ML baseline
  ridge   - linear regression on all sequence + thermodynamic features
  forest  - gradient-boosted trees on the same features

Run from the project root:  python -m src.train
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import DATA_PATH, dataset_summary, load_dataset
from src.features import feature_names, featurize_many, reynolds_score

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
MODEL_FILE = "sirna_model.pkl"
ACTIVE_THRESHOLD = 0.7  # >=70% knockdown counts as an active siRNA
N_SPLITS = 5
RANDOM_STATE = 42


def build_models():
    return {
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
        "forest": HistGradientBoostingRegressor(
            max_depth=6, learning_rate=0.06, max_iter=400,
            early_stopping=True, validation_fraction=0.15, random_state=RANDOM_STATE),
    }


def score(y_true, y_pred, threshold=ACTIVE_THRESHOLD):
    """Correlation and error, plus how well the model separates active siRNAs."""
    result = {
        "pearson_r": float(pearsonr(y_true, y_pred)[0]),
        "spearman_r": float(spearmanr(y_true, y_pred)[0]),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
    active = (np.asarray(y_true) >= threshold).astype(int)
    result["auc"] = float(roc_auc_score(active, y_pred)) if 0 < active.sum() < len(active) else None
    return result


def cross_validate(X, y, groups, rule_scores, n_splits=N_SPLITS):
    """Grouped CV: every fold tests on genes that were not in its training set."""
    n_splits = min(n_splits, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=n_splits)
    predictions = {name: np.zeros(len(y)) for name in list(build_models()) + ["rules"]}

    for train_idx, test_idx in splitter.split(X, y, groups):
        predictions["rules"][test_idx] = rule_scores[test_idx]
        for name, model in build_models().items():
            model.fit(X[train_idx], y[train_idx])
            predictions[name][test_idx] = model.predict(X[test_idx])

    return {name: score(y, pred) for name, pred in predictions.items()}, predictions


def per_target_scores(df, predictions):
    """Where the model does well and badly, gene by gene."""
    rows = []
    for target, group in df.assign(pred=predictions).groupby("target"):
        if len(group) < 10 or group["efficacy"].nunique() < 3:
            continue
        rows.append({"target": target, "sirnas": int(len(group)),
                     "pearson_r": round(float(pearsonr(group["efficacy"], group["pred"])[0]), 3),
                     "mean_efficacy": round(float(group["efficacy"].mean()), 3)})
    return sorted(rows, key=lambda r: -r["pearson_r"])


def train(data_path=DATA_PATH, model_dir=MODEL_DIR, n_splits=N_SPLITS):
    df = load_dataset(data_path)
    X = featurize_many(df["guide"])
    y = df["efficacy"].to_numpy()
    groups = df["target"].to_numpy()
    rule_scores = np.array([reynolds_score(g) for g in df["guide"]], dtype=float)

    cv_scores, predictions = cross_validate(X, y, groups, rule_scores, n_splits)
    best_name = max((n for n in cv_scores if n != "rules"),
                    key=lambda n: cv_scores[n]["pearson_r"])
    best_model = build_models()[best_name].fit(X, y)  # refit on everything for serving

    names = feature_names()
    if best_name == "ridge":
        weights = best_model[-1].coef_
    else:
        from sklearn.inspection import permutation_importance
        weights = permutation_importance(best_model, X, y, n_repeats=5,
                                         random_state=RANDOM_STATE).importances_mean
    top_features = sorted(zip(names, (float(abs(w)) for w in weights)),
                          key=lambda t: -t[1])[:15]

    trained_at = datetime.now(timezone.utc)
    info = {
        "version": trained_at.strftime("%Y%m%d-%H%M%S"),
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "best_model": best_name,
        "validation": f"GroupKFold by target gene, {min(n_splits, df['target'].nunique())} folds",
        "active_threshold": ACTIVE_THRESHOLD,
        "data": dataset_summary(df, ACTIVE_THRESHOLD),
        "cv_scores": {name: {k: (round(v, 4) if v is not None else None)
                             for k, v in s.items()} for name, s in cv_scores.items()},
        "top_features": [{"feature": f, "weight": round(w, 4)} for f, w in top_features],
        "per_target": per_target_scores(df, predictions[best_name]),
        "sklearn_version": sklearn.__version__,
    }

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best_model, "info": info}, model_dir / MODEL_FILE)
    (model_dir / "metrics.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    df.to_csv(model_dir / "training_data.csv", index=False)
    return info


if __name__ == "__main__":
    result = train()
    d = result["data"]
    print(f"{d['sirnas']} siRNAs across {d['targets']} target genes "
          f"({d['active_fraction']:.0%} active at >={d['active_threshold']:.0%} knockdown)")
    print(f"Validation: {result['validation']}\n")
    print(pd.DataFrame(result["cv_scores"]).T.to_string())
    print(f"\nBest model: {result['best_model']}")
    print("Top features:", ", ".join(f["feature"] for f in result["top_features"][:8]))
    print(f"Saved to {MODEL_DIR / MODEL_FILE}")

# Amrabad risk pipeline: run after downloading the GEE CSV
# pip install pandas numpy scikit-learn xgboost shap matplotlib
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

CSV = "data/amrabad_2023_2026raw.csv"   # change to your file name
HORIZON = 2                         # label = anomaly this many months ahead
THRESH = -1.0                       # anomaly threshold (SD); test -0.5 and -1.5 too

# ---------- 1. Load and clean ----------
df = pd.read_csv(CSV)
df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))
df = df.replace([np.inf, -np.inf], np.nan)

print("raw pixels:", df.pixel_id.nunique(), "| raw rows:", len(df))

# rebuild the full pixel x month grid, then fill short gaps (monsoon cloud masking)
full = pd.MultiIndex.from_product(
    [df.pixel_id.unique(), sorted(df.date.unique())], names=["pixel_id", "date"])
df = df.set_index(["pixel_id", "date"]).reindex(full).reset_index()
df["year"], df["month"] = df.date.dt.year, df.date.dt.month

static = ["longitude", "latitude", "elevation", "slope", "aspect"]
df[static] = df.groupby("pixel_id")[static].transform("first")

dyn = ["ndvi", "ndvi_anom", "ndwi", "rain", "lst"]
miss = df[dyn].isna().mean().round(3)
print("share missing before fill:\n", miss.to_string())
df = df.sort_values(["pixel_id", "date"])
df[dyn] = df.groupby("pixel_id")[dyn].transform(
    lambda s: s.interpolate(limit=2, limit_direction="both"))

# drop only pixels that still have gaps after filling
bad = df[df[dyn].isna().any(axis=1)].pixel_id.unique()
df = df[~df.pixel_id.isin(bad)].copy()
print("pixels kept:", df.pixel_id.nunique(), "of", len(bad) + df.pixel_id.nunique())

df = df.sort_values(["pixel_id", "date"]).reset_index(drop=True)
g = df.groupby("pixel_id")

# ---------- 2. Lag / rolling features ----------
for lag in [1, 2, 3]:
    df[f"rain_lag{lag}"] = g["rain"].shift(lag)
    df[f"lst_lag{lag}"] = g["lst"].shift(lag)
    df[f"anom_lag{lag}"] = g["ndvi_anom"].shift(lag)
df["rain_3m"] = g["rain"].transform(lambda s: s.rolling(3).sum())
df["lst_3m"] = g["lst"].transform(lambda s: s.rolling(3).mean())
df["m_sin"] = np.sin(2 * np.pi * df.month / 12)
df["m_cos"] = np.cos(2 * np.pi * df.month / 12)

# spatial block id (~5 km blocks) for spatial CV later
df["block_id"] = (df.latitude // 0.05).astype(int).astype(str) + "_" + (df.longitude // 0.05).astype(int).astype(str)

# ---------- 3. Label (future anomaly) ----------
df["future_anom"] = g["ndvi_anom"].shift(-HORIZON)
df["label"] = np.where(df.future_anom.isna(), np.nan, (df.future_anom < THRESH).astype(float))

ndvi_feats = ["ndvi", "ndvi_anom", "anom_lag1", "anom_lag2", "anom_lag3"]
other_feats = ["ndwi", "rain", "lst", "elevation", "slope", "aspect",
               "rain_lag1", "rain_lag2", "rain_lag3", "lst_lag1", "lst_lag2", "lst_lag3",
               "rain_3m", "lst_3m", "m_sin", "m_cos"]
features = ndvi_feats + other_feats

df = df.dropna(subset=features)        # drops the 3 lead-in months

# ---------- 4. Temporal split ----------
labeled = df.dropna(subset=["label"])
train = labeled[labeled.year <= 2024]
val = labeled[labeled.year == 2025]
test = labeled[labeled.year == 2026]
last = df.date.max()
predict = df[df.date >= last - pd.DateOffset(months=HORIZON - 1)]   # unlabeled, latest months
print(f"train {len(train)} | val {len(val)} | test {len(test)} | predict {len(predict)}")
print("positive rate train/val/test:", train.label.mean().round(3), val.label.mean().round(3), test.label.mean().round(3))

# ---------- 5. Models ----------
def report(name, y, p):
    pred = (p >= 0.3).astype(int)
    print(f"{name}: PR-AUC {average_precision_score(y, p):.3f} | ROC-AUC {roc_auc_score(y, p):.3f} "
          f"| F1 {f1_score(y, pred):.3f} | recall {recall_score(y, pred):.3f} | base rate {y.mean():.3f}")

pos_w = (train.label == 0).sum() / (train.label == 1).sum()
xgb = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, scale_pos_weight=pos_w, eval_metric="aucpr", random_state=42)
xgb.fit(train[features], train.label, eval_set=[(val[features], val.label)], verbose=False)

lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
lr.fit(train[features], train.label)

for name, d in [("VAL 2025", val), ("TEST 2026", test)]:
    print(name)
    report("  XGBoost", d.label, xgb.predict_proba(d[features])[:, 1])
    report("  LogReg ", d.label, lr.predict_proba(d[features])[:, 1])

# Model without NDVI-based features (external drivers only)
xgb2 = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                     colsample_bytree=0.8, scale_pos_weight=pos_w, eval_metric="aucpr", random_state=42)
xgb2.fit(train[other_feats], train.label)
print("TEST 2026, no NDVI features")
report("  XGBoost", test.label, xgb2.predict_proba(test[other_feats])[:, 1])

# ---------- 5b. Results table for the README ----------
import os
from sklearn.metrics import precision_recall_curve

def pick_threshold(y, p):
    """Cutoff that maximises F1 on VALIDATION data."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return thr[np.argmax(f1[:-1])]

def score_row(name, p_val, p_test):
    t = pick_threshold(val.label, p_val)          # threshold chosen on val only
    pred = (p_test >= t).astype(int)
    return {"Model": name,
            "PR-AUC": average_precision_score(test.label, p_test),
            "ROC-AUC": roc_auc_score(test.label, p_test),
            "F1": f1_score(test.label, pred),
            "Recall": recall_score(test.label, pred)}

rows = [
    score_row("Persistence baseline", -val.ndvi_anom, -test.ndvi_anom),
    score_row("Logistic Regression",
              lr.predict_proba(val[features])[:, 1], lr.predict_proba(test[features])[:, 1]),
    score_row("XGBoost (external features only, no NDVI)",
              xgb2.predict_proba(val[other_feats])[:, 1], xgb2.predict_proba(test[other_feats])[:, 1]),
    score_row("XGBoost (all features)",
              xgb.predict_proba(val[features])[:, 1], xgb.predict_proba(test[features])[:, 1]),
]
results = pd.DataFrame(rows).round(3)
os.makedirs("outputs", exist_ok=True)
results.to_csv("outputs/metrics.csv", index=False)

print("Test prevalence:", round(test.label.mean(), 3))
print("| " + " | ".join(results.columns) + " |")
print("|" + "---|" * len(results.columns))
for _, r in results.iterrows():
    print("| " + " | ".join(str(v) for v in r) + " |")

# Do the risk classes mean anything? Loss rate should rise Low -> Very High
cat = pd.cut(xgb.predict_proba(test[features])[:, 1], [0, .10, .20, .30, 1.0],
             labels=["Low", "Moderate", "High", "Very High"], include_lowest=True)
print(test.groupby(cat, observed=True).label.agg(["mean", "count"]))

# ---------- 6. SHAP ----------
import shap, matplotlib.pyplot as plt
sample = test[features].sample(min(5000, len(test)), random_state=1)
sv = shap.TreeExplainer(xgb).shap_values(sample)
shap.summary_plot(sv, sample, show=False); plt.tight_layout(); plt.savefig("outputs/shap_all.png", dpi=150); plt.close()
sample2 = test[other_feats].sample(min(5000, len(test)), random_state=1)
sv2 = shap.TreeExplainer(xgb2).shap_values(sample2)
shap.summary_plot(sv2, sample2, show=False); plt.tight_layout(); plt.savefig("outputs/shap_external_only.png", dpi=150); plt.close()

# ---------- 7. Risk table for the map ----------
out = predict[["pixel_id", "date", "longitude", "latitude"]].copy()
out["risk"] = xgb.predict_proba(predict[features])[:, 1]
out["risk_class"] = pd.cut(
    out.risk,
    [0, .10, .20, .30, 1.0],
    labels=["Low", "Moderate", "High", "Very High"],
    include_lowest=True
)
out.to_csv("outputs/risk_predictions.csv", index=False)
print(out.groupby("date").risk_class.value_counts().unstack())
print("saved: risk_predictions.csv, shap_all.png, shap_external_only.png")


# ---------- 8. Export everything the dashboard needs ----------
import os, json, shutil
os.makedirs("dashboard_data", exist_ok=True)

panel = df.copy()
panel["risk"] = xgb.predict_proba(panel[features])[:, 1]
panel["split"] = np.select(
    [panel.label.isna(), panel.year <= 2024, panel.year == 2025],
    ["predict", "train", "val"], default="test")

# Risk classes are relative to the 2025-2026 score distribution (it is an index, not a probability)
thr = {
    "moderate": 0.10,
    "high": 0.20,
    "very_high": 0.30
}

def evaluate(d):
    top = d[d.risk >= d.risk.quantile(0.90)]
    return {"rows": int(len(d)), "base_rate": float(d.label.mean()),
            "pr_auc": float(average_precision_score(d.label, d.risk)),
            "roc_auc": float(roc_auc_score(d.label, d.risk)),
            "top10_hit_rate": float(top.label.mean()),
            "lift_top10": float(top.label.mean() / d.label.mean())}

p2 = xgb2.predict_proba(test[other_feats])[:, 1]
meta = {"features": features, "ndvi_feats": ndvi_feats, "other_feats": other_feats,
        "horizon_months": HORIZON, "threshold_sd": THRESH, "thresholds": thr,
        "n_pixels": int(panel.pixel_id.nunique()), "last_date": str(df.date.max().date()),
        "val": evaluate(panel[panel.split == "val"]),
        "test": evaluate(panel[panel.split == "test"]),
        "test_no_ndvi": {"pr_auc": float(average_precision_score(test.label, p2)),
                         "roc_auc": float(roc_auc_score(test.label, p2))}}

keep = ["pixel_id", "date", "year", "month", "longitude", "latitude", "block_id",
        "split", "label", "future_anom", "risk"] + features
panel[keep].to_csv("dashboard_data/panel.csv.gz", index=False)
xgb.save_model("dashboard_data/model.json")
json.dump(meta, open("dashboard_data/metrics.json", "w"), indent=2)
for f in ["shap_all.png", "shap_external_only.png"]:
    shutil.copy(f, "dashboard_data/" + f)
print("TOP-10% LIFT  val:", round(meta["val"]["lift_top10"], 2), "| test:", round(meta["test"]["lift_top10"], 2))
print("dashboard_data/ ready: now run  streamlit run app.py")
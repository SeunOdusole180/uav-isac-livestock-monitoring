import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
print("START results_tables_graphs.py")
BASE = Path("/home/vboxuser/ns3-workspace/ns-3-dev")
run_summary_path = BASE / "run_summary.csv"
feature_imp_path = BASE / "offline_feature_importance.csv"
ranked_runs_path = BASE / "offline_ranked_runs.csv"
suggested_path = BASE / "offline_suggested_candidates.csv"

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(run_summary_path)
feat = pd.read_csv(feature_imp_path)
ranked = pd.read_csv(ranked_runs_path)

suggested = None
if suggested_path.exists():
    suggested = pd.read_csv(suggested_path)

print(f"Loaded {len(df)} runs from run_summary.csv")

# ----------------------------
# Define useful subsets
# ----------------------------
df = df.sort_values("rngRun").reset_index(drop=True)

# Early baseline set = first 40 original exploration runs
baseline_df = df[df["rngRun"] <= 40].copy()
candidate_df = df[df["rngRun"] > 40].copy()

best_baseline = baseline_df.loc[baseline_df["Jfinal"].idxmax()] if len(baseline_df) else None
best_overall = df.loc[df["Jfinal"].idxmax()]
best_energy = df.loc[df["EtotJ"].idxmin()]
best_rmse = df.loc[df["RMSEm"].idxmin()]

# Top 5 overall
top5 = df.sort_values("Jfinal", ascending=False).head(5).copy()

# ----------------------------
# Table 1: Main performance comparison
# ----------------------------
rows = []

if best_baseline is not None:
    rows.append({
        "Configuration": "Best baseline (runs 1-40)",
        "rngRun": int(best_baseline["rngRun"]),
        "Pdet": best_baseline["Pdet"],
        "RMSEm": best_baseline["RMSEm"],
        "avgDelayMs": best_baseline["avgDelayMs"],
        "avgThrMbps": best_baseline["avgThrMbps"],
        "EtotJ": best_baseline["EtotJ"],
        "Jfinal": best_baseline["Jfinal"],
    })

rows.append({
    "Configuration": "Best overall",
    "rngRun": int(best_overall["rngRun"]),
    "Pdet": best_overall["Pdet"],
    "RMSEm": best_overall["RMSEm"],
    "avgDelayMs": best_overall["avgDelayMs"],
    "avgThrMbps": best_overall["avgThrMbps"],
    "EtotJ": best_overall["EtotJ"],
    "Jfinal": best_overall["Jfinal"],
})

rows.append({
    "Configuration": "Lowest RMSE",
    "rngRun": int(best_rmse["rngRun"]),
    "Pdet": best_rmse["Pdet"],
    "RMSEm": best_rmse["RMSEm"],
    "avgDelayMs": best_rmse["avgDelayMs"],
    "avgThrMbps": best_rmse["avgThrMbps"],
    "EtotJ": best_rmse["EtotJ"],
    "Jfinal": best_rmse["Jfinal"],
})

rows.append({
    "Configuration": "Lowest energy",
    "rngRun": int(best_energy["rngRun"]),
    "Pdet": best_energy["Pdet"],
    "RMSEm": best_energy["RMSEm"],
    "avgDelayMs": best_energy["avgDelayMs"],
    "avgThrMbps": best_energy["avgThrMbps"],
    "EtotJ": best_energy["EtotJ"],
    "Jfinal": best_energy["Jfinal"],
})

table_main = pd.DataFrame(rows)
table_main.to_csv(BASE / "results_table_main.csv", index=False)

# ----------------------------
# Table 2: Best learned policy
# ----------------------------
policy_cols = [
    "survSenseDt","patSenseDt","rapidSenseDt","stratSenseDt",
    "alphaSurv","alphaPat","alphaRapid","alphaStrat",
    "survVmin","survVmax","patVmin","patVmax",
    "rapidVmin","rapidVmax","stratVmin","stratVmax",
    "survZmin","survZmax","patZmin","patZmax",
    "rapidZmin","rapidZmax","stratZmin","stratZmax",
    "survDt","patDt","rapidDt","stratDt"
]

policy_table = pd.DataFrame({
    "Variable": policy_cols,
    "BestValue": [best_overall[c] for c in policy_cols]
})
policy_table.to_csv(BASE / "results_table_best_policy.csv", index=False)

# ----------------------------
# Table 3: Top 5 runs
# ----------------------------
top5.to_csv(BASE / "results_table_top5_runs.csv", index=False)

# ----------------------------
# Table 4: Feature importance
# ----------------------------
feat_sorted = feat.sort_values(feat.columns[-1], ascending=False).head(10).copy()
feat_sorted.to_csv(BASE / "results_table_feature_importance_top10.csv", index=False)

# ----------------------------
# Figure 1: Optimisation progress
# ----------------------------
plt.figure(figsize=(9, 5))
plt.plot(df["rngRun"], df["Jfinal"], marker="o")
plt.xlabel("Run number")
plt.ylabel("Jfinal")
plt.title("Optimisation progress across ns-3 runs")
plt.grid(True)
plt.tight_layout()
plt.savefig(BASE / "fig_optimisation_progress.png", dpi=300)
plt.close()

# ----------------------------
# Figure 2: Baseline vs best overall
# ----------------------------
if best_baseline is not None:
    compare = pd.DataFrame([
        ["Best baseline", best_baseline["Pdet"], best_baseline["RMSEm"], best_baseline["EtotJ"], best_baseline["Jfinal"]],
        ["Best overall", best_overall["Pdet"], best_overall["RMSEm"], best_overall["EtotJ"], best_overall["Jfinal"]],
    ], columns=["Config", "Pdet", "RMSEm", "EtotJ", "Jfinal"])

    metrics = ["Pdet", "RMSEm", "EtotJ", "Jfinal"]
    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width/2, compare.iloc[0, 1:], width, label=compare.iloc[0, 0])
    plt.bar(x + width/2, compare.iloc[1, 1:], width, label=compare.iloc[1, 0])
    plt.xticks(x, metrics)
    plt.ylabel("Value")
    plt.title("Baseline vs optimised performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE / "fig_baseline_vs_best.png", dpi=300)
    plt.close()

# ----------------------------
# Figure 3: Feature importance
# ----------------------------
importance_col = feat_sorted.columns[-1]
name_col = feat_sorted.columns[0]

plt.figure(figsize=(8, 5))
plt.barh(feat_sorted[name_col][::-1], feat_sorted[importance_col][::-1])
plt.xlabel("Importance score")
plt.ylabel("Parameter")
plt.title("Top 10 most important optimisation variables")
plt.tight_layout()
plt.savefig(BASE / "fig_feature_importance.png", dpi=300)
plt.close()

# ----------------------------
# Figure 4: Energy vs RMSE trade-off
# ----------------------------
plt.figure(figsize=(8, 5))
sc = plt.scatter(df["EtotJ"], df["RMSEm"], c=df["Jfinal"])
plt.xlabel("Total energy (J)")
plt.ylabel("RMSE (m)")
plt.title("Energy–accuracy trade-off across runs")
plt.colorbar(sc, label="Jfinal")
plt.tight_layout()
plt.savefig(BASE / "fig_energy_vs_rmse_tradeoff.png", dpi=300)
plt.close()

# ----------------------------
# Figure 5: Jfinal vs Energy with run index and best runs highlighted
# ----------------------------
plt.figure(figsize=(8, 5))
sc = plt.scatter(df["EtotJ"], df["Jfinal"], c=df["rngRun"])
plt.xlabel("Total energy (J)")
plt.ylabel("Jfinal")
plt.title("Objective score versus energy across runs")
plt.colorbar(sc, label="Run number")

# label best few runs
top_runs = df.sort_values("Jfinal", ascending=False).head(5)
for _, row in top_runs.iterrows():
    plt.annotate(
        f"Run {int(row['rngRun'])}",
        (row["EtotJ"], row["Jfinal"]),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=8
    )

plt.tight_layout()
plt.savefig(BASE / "fig_jfinal_vs_energy_runs.png", dpi=300)
plt.close()

# ----------------------------
# Figure 7: Baseline vs best overall (normalized to baseline)
# ----------------------------
if best_baseline is not None:
    baseline_vals = {
        "Pdet": best_baseline["Pdet"],
        "RMSEm": best_baseline["RMSEm"],
        "EtotJ": best_baseline["EtotJ"],
        "Jfinal": best_baseline["Jfinal"],
    }
    best_vals = {
        "Pdet": best_overall["Pdet"],
        "RMSEm": best_overall["RMSEm"],
        "EtotJ": best_overall["EtotJ"],
        "Jfinal": best_overall["Jfinal"],
    }

    metrics = ["Pdet", "RMSEm", "EtotJ", "Jfinal"]
    norm_baseline = [1.0 for _ in metrics]
    norm_best = [best_vals[m] / baseline_vals[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, norm_baseline, width, label=f"Baseline run {int(best_baseline['rngRun'])}")
    plt.bar(x + width/2, norm_best, width, label=f"Best run {int(best_overall['rngRun'])}")
    plt.xticks(x, metrics)
    plt.ylabel("Normalized value (baseline = 1)")
    plt.title("Baseline versus best run (normalized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE / "fig_baseline_vs_best_normalized.png", dpi=300)
    plt.close()

    # ----------------------------
# Table: top runs compared to best baseline
# ----------------------------
best_baseline = df[df["rngRun"] <= 40].sort_values("Jfinal", ascending=False).iloc[0]
top_runs = df.sort_values("Jfinal", ascending=False).head(5).copy()

comparison_rows = []
for _, row in top_runs.iterrows():
    delta_j = row["Jfinal"] - best_baseline["Jfinal"]
    delta_e = row["EtotJ"] - best_baseline["EtotJ"]
    comparison_rows.append({
        "Run": int(row["rngRun"]),
        "Jscore": row["Jfinal"],
        "EnergyJ": row["EtotJ"],
        "Delta_J_vs_Baseline": delta_j,
        "Delta_EnergyJ_vs_Baseline": delta_e,
        "Delta_J_per_kJ": delta_j / (delta_e / 1000.0) if abs(delta_e) > 1e-9 else np.nan
    })

delta_table = pd.DataFrame(comparison_rows)
delta_table.to_csv(BASE / "results_table_improvement_per_energy.csv", index=False)

# ----------------------------
# Figure: RMSE before vs after optimisation
# ----------------------------
baseline = df[df["rngRun"] <= 40]["RMSEm"]
post_opt = df[df["rngRun"] >= 41]["RMSEm"]

plt.figure(figsize=(7, 5))
plt.boxplot([baseline, post_opt], labels=["Runs 1-40", "Runs 41-54"])
plt.ylabel("RMSE (m)")
plt.title("Localization accuracy before and after optimisation")
plt.tight_layout()
plt.savefig(BASE / "fig_rmse_before_after.png", dpi=300)
plt.close()


# ----------------------------
# Figure 8: RMSE vs Run
# ----------------------------
plt.figure(figsize=(9, 5))
plt.plot(df["rngRun"], df["RMSEm"], marker="o")
plt.xlabel("Run number")
plt.ylabel("RMSE (m)")
plt.title("Localisation accuracy across simulation runs")
plt.grid(True)
plt.tight_layout()
plt.savefig(BASE / "fig_rmse_vs_run.png", dpi=300)
plt.close()

print("Saved:")
print(" - results_table_main.csv")
print(" - results_table_best_policy.csv")
print(" - results_table_top5_runs.csv")
print(" - results_table_feature_importance_top10.csv")
print(" - fig_optimisation_progress.png")
print(" - fig_baseline_vs_best.png")
print(" - fig_feature_importance.png")
print(" - fig_energy_vs_rmse_tradeoff.png")
print(" - fig_jfinal_vs_energy_runs.png")
print(" - fig_baseline_vs_best_normalized.png")
print("DONE")
"""
results_analysis_thesis_complete.py
====================================
Complete thesis results — tables and figures from:
  - run_summary.csv       (2D optimisation dataset)
  - run_summary_3d.csv    (3D extended dataset)
  - xapp_kpi_log.csv      (xApp near-RT RIC KPI tracking)

Run: python scratch/results_analysis_thesis_complete.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────
BASE = Path("/home/vboxuser/ns3-workspace/ns-3-dev")
OUT  = BASE / "thesis_figures"
OUT.mkdir(exist_ok=True)

# ── Load data ────────────────────────────────────────────────
print("Loading data...")
df2d = pd.read_csv(BASE / "run_summary.csv").sort_values("rngRun").reset_index(drop=True)
df3d = pd.read_csv(BASE / "run_summary_3d.csv").sort_values("rngRun").reset_index(drop=True)
xkpi = pd.read_csv(BASE / "xapp_kpi_log.csv")

feat_path = BASE / "offline_feature_importance.csv"
feat = pd.read_csv(feat_path) if feat_path.exists() else None

print(f"  2D runs  : {len(df2d)}")
print(f"  3D runs  : {len(df3d)}")
print(f"  xApp KPIs: {len(xkpi)} windows")

# ── Shared config ────────────────────────────────────────────
C     = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
ROLES = ["Surveillance", "Patrol", "Rapid-Response", "Strategic"]
FSAVE = dict(dpi=150, bbox_inches="tight")

# Phase splits
base2d = df2d[df2d["rngRun"] <= 50]
post2d = df2d[df2d["rngRun"] >  50]
best_off_2d = df2d.loc[df2d["Jfinal"].idxmax()]

base3d = df3d[df3d["rngRun"] <= 50]
post3d = df3d[df3d["rngRun"] >  50]
best_off_3d = df3d.loc[df3d["Jfinal"].idxmax()]

top20_2d = df2d.sort_values("Jfinal", ascending=False).head(20)
top20_3d = df3d.sort_values("Jfinal", ascending=False).head(20)

# xApp summary
xapp_pdet   = xkpi["pdet"].iloc[-1]
xapp_rmse   = xkpi["rmse_m"].iloc[-1]
xapp_jscore = xkpi["jscore"].iloc[-1]
xapp_energy = xkpi["energy_j"].iloc[-1]

t = xkpi["sim_time_s"]

saved = []

def save(name):
    plt.savefig(OUT / name, **FSAVE)
    plt.close()
    saved.append(name)
    print(f"  Saved: {name}")

from matplotlib.patches import Patch

# ============================================================
# ██████████████  TABLES  ████████████████████████████████████
# ============================================================
print("\n── Generating tables ──")

def styled_table(df, title, fname, col_formats=None):
    """Save a DataFrame as a styled PNG table."""
    fig, ax = plt.subplots(figsize=(max(8, len(df.columns)*1.8),
                                    max(3, len(df)*0.4 + 1.2)))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.auto_set_column_width(col=list(range(len(df.columns))))
    # Header styling
    for j in range(len(df.columns)):
        tbl[(0, j)].set_facecolor("#2c5f8a")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    # Alternating row colours
    for i in range(1, len(df)+1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                tbl[(i, j)].set_facecolor("#f0f4f8")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=12)
    plt.tight_layout()
    save(fname)


# ── TABLE T1: Before/After Optimisation (2D) ────────────────
t1 = pd.DataFrame([
    ["Baseline exploration", "1–50",
     f"{base2d['Jfinal'].mean():.3f}",
     f"{base2d['Pdet'].mean():.3f}",
     f"{base2d['RMSEm'].mean():.2f}",
     f"{base2d['EtotJ'].mean():.0f}"],
    ["RL-Guided (post-opt)", f"51–{df2d['rngRun'].max()}",
     f"{post2d['Jfinal'].mean():.3f}",
     f"{post2d['Pdet'].mean():.3f}",
     f"{post2d['RMSEm'].mean():.2f}",
     f"{post2d['EtotJ'].mean():.0f}"],
], columns=["Phase","Runs","Avg Jscore","Avg Pdet","Avg RMSE (m)","Avg Energy (J)"])
styled_table(t1,
    "Table T1 — System Performance Before and After Optimisation (2D)",
    "T1_performance_before_after_2d.png")

# ── TABLE T2: Before/After Optimisation (3D) ────────────────
t2 = pd.DataFrame([
    ["Baseline exploration", "1–50",
     f"{base3d['Jfinal'].mean():.3f}",
     f"{base3d['Pdet'].mean():.3f}",
     f"{base3d['RMSEm_3D'].mean():.2f}",
     f"{base3d['EtotJ'].mean():.0f}"],
    ["RL-Guided (post-opt)", f"51–{df3d['rngRun'].max()}",
     f"{post3d['Jfinal'].mean():.3f}",
     f"{post3d['Pdet'].mean():.3f}",
     f"{post3d['RMSEm_3D'].mean():.2f}",
     f"{post3d['EtotJ'].mean():.0f}"],
], columns=["Phase","Runs","Avg Jscore","Avg Pdet","Avg RMSE (m)","Avg Energy (J)"])
styled_table(t2,
    "Table T2 — System Performance Before and After Optimisation (3D)",
    "T2_performance_before_after_3d.png")

# ── TABLE T3: Localisation accuracy start vs end (2D) ───────
early2d = df2d[df2d["rngRun"] <= 10][["rngRun","Jfinal","RMSEm"]].copy()
early2d["Phase"] = "Initial exploration"
late2d  = df2d.sort_values("Jfinal", ascending=False).head(10)[
    ["rngRun","Jfinal","RMSEm"]].copy()
late2d["Phase"] = "Final optimisation"
t3 = pd.concat([early2d, late2d], ignore_index=True)
t3 = t3[["rngRun","Phase","Jfinal","RMSEm"]]
t3.columns = ["Run","Phase","Jscore","RMSE (m)"]
t3["Jscore"] = t3["Jscore"].apply(lambda x: f"{x:.5f}")
t3["RMSE (m)"] = t3["RMSE (m)"].apply(lambda x: f"{x:.2f}")
# Add means row
means_row = pd.DataFrame([
    ["Mean (1–10)", "—",
     f"{early2d['Jfinal'].mean():.2f}",
     f"{early2d['RMSEm'].mean():.2f}"],
    [f"Mean (top 10)", "—",
     f"{late2d['Jfinal'].mean():.2f}",
     f"{late2d['RMSEm'].mean():.2f}"],
], columns=t3.columns)
t3_full = pd.concat([t3, means_row], ignore_index=True)
styled_table(t3_full,
    "Table T3 — Localisation Accuracy at Start and End of Optimisation (2D)",
    "T3_localisation_start_end_2d.png")

# ── TABLE T4: Localisation accuracy start vs end (3D) ───────
early3d = df3d[df3d["rngRun"] <= 10][["rngRun","Jfinal","RMSEm_3D"]].copy()
early3d["Phase"] = "Initial exploration"
late3d  = df3d.sort_values("Jfinal", ascending=False).head(10)[
    ["rngRun","Jfinal","RMSEm_3D"]].copy()
late3d["Phase"] = "Final optimisation"
t4 = pd.concat([early3d, late3d], ignore_index=True)
t4 = t4[["rngRun","Phase","Jfinal","RMSEm_3D"]]
t4.columns = ["Run","Phase","Jscore","RMSE (m)"]
t4["Jscore"]   = t4["Jscore"].apply(lambda x: f"{x:.5f}")
t4["RMSE (m)"] = t4["RMSE (m)"].apply(lambda x: f"{x:.2f}")
means_3d = pd.DataFrame([
    ["Mean (1–10)", "—",
     f"{early3d['Jfinal'].mean():.2f}",
     f"{early3d['RMSEm_3D'].mean():.2f}"],
    ["Mean (top 10)", "—",
     f"{late3d['Jfinal'].mean():.2f}",
     f"{late3d['RMSEm_3D'].mean():.2f}"],
], columns=t4.columns)
t4_full = pd.concat([t4, means_3d], ignore_index=True)
styled_table(t4_full,
    "Table T4 — Localisation Accuracy at Start and End of Optimisation (3D)",
    "T4_localisation_start_end_3d.png")

# ── TABLE T5: Performance improvement vs baseline (2D) ──────
best_base_2d = base2d.loc[base2d["Jfinal"].idxmax()]
top5_2d = post2d.sort_values("Jfinal", ascending=False).head(5)
rows_t5 = [{
    "Run": f"{int(best_base_2d['rngRun'])} (Baseline)",
    "Jscore": f"{best_base_2d['Jfinal']:.5f}",
    "Energy (J)": f"{best_base_2d['EtotJ']:.0f}",
    "ΔJ": "0",
    "ΔE (J)": "0",
    "ΔJ/kJ": "0"
}]
for _, row in top5_2d.iterrows():
    dj = row["Jfinal"] - best_base_2d["Jfinal"]
    de = row["EtotJ"] - best_base_2d["EtotJ"]
    rows_t5.append({
        "Run": int(row["rngRun"]),
        "Jscore": f"{row['Jfinal']:.5f}",
        "Energy (J)": f"{row['EtotJ']:.0f}",
        "ΔJ": f"{dj:.5f}",
        "ΔE (J)": f"{de:.0f}",
        "ΔJ/kJ": f"{dj/(de/1000):.4f}" if abs(de) > 1 else "—"
    })
t5 = pd.DataFrame(rows_t5)
styled_table(t5,
    "Table T5 — Performance Improvement Relative to Best Baseline (2D)",
    "T5_improvement_vs_baseline_2d.png")

# ── TABLE T6: Performance improvement vs baseline (3D) ──────
best_base_3d = base3d.loc[base3d["Jfinal"].idxmax()]
top5_3d = post3d.sort_values("Jfinal", ascending=False).head(5)
rows_t6 = [{
    "Run": f"{int(best_base_3d['rngRun'])} (Baseline)",
    "Jscore": f"{best_base_3d['Jfinal']:.5f}",
    "Energy (J)": f"{best_base_3d['EtotJ']:.0f}",
    "ΔJ": "0", "ΔE (J)": "0", "ΔJ/kJ": "0"
}]
for _, row in top5_3d.iterrows():
    dj = row["Jfinal"] - best_base_3d["Jfinal"]
    de = row["EtotJ"] - best_base_3d["EtotJ"]
    rows_t6.append({
        "Run": int(row["rngRun"]),
        "Jscore": f"{row['Jfinal']:.5f}",
        "Energy (J)": f"{row['EtotJ']:.0f}",
        "ΔJ": f"{dj:.5f}",
        "ΔE (J)": f"{de:.0f}",
        "ΔJ/kJ": f"{dj/(de/1000):.4f}" if abs(de) > 1 else "—"
    })
t6 = pd.DataFrame(rows_t6)
styled_table(t6,
    "Table T6 — Performance Improvement Relative to Best Baseline (3D)",
    "T6_improvement_vs_baseline_3d.png")

# ── TABLE T7: Three-way comparison summary ──────────────────
t7 = pd.DataFrame([
    ["Baseline mean (2D)", "1–50",
     f"{base2d['Pdet'].mean():.4f}",
     f"{base2d['RMSEm'].mean():.2f}",
     f"{base2d['Jfinal'].mean():.4f}",
     f"{base2d['EtotJ'].mean()/1000:.1f}",
     f"{base2d['avgThrMbps'].mean()*1000:.3f}",
     f"{base2d['avgDelayMs'].mean():.2f}"],
    ["Best offline (2D)", f"Run {int(best_off_2d['rngRun'])}",
     f"{best_off_2d['Pdet']:.4f}",
     f"{best_off_2d['RMSEm']:.2f}",
     f"{best_off_2d['Jfinal']:.4f}",
     f"{best_off_2d['EtotJ']/1000:.1f}",
     f"{best_off_2d['avgThrMbps']*1000:.3f}",
     f"{best_off_2d['avgDelayMs']:.2f}"],
    ["xApp", "300s run",
     f"{xapp_pdet:.4f}",
     f"{xapp_rmse:.2f}",
     f"{xapp_jscore:.4f}",
     f"{xapp_energy/1000:.1f}",
     f"{xkpi['avg_throughput_mbps'].mean()*1000:.3f}",
     f"{xkpi['avg_delay_ms'].mean():.2f}"],
], columns=["Configuration","Phase/Run","Pdet","RMSE (m)",
            "Jscore","Energy (kJ)","Thr (kbps)","Delay (ms)"])
styled_table(t7,
    "Table T7 — Three-Way Performance Summary: Baseline vs Offline RL vs xApp",
    "T7_three_way_summary.png")

# ── TABLE T8: Per-role summary (3D top 20) ──────────────────
role_data = []
for role, det, rate, rmse, cov, duty, epd in zip(
    ROLES,
    ["survDetections","patDetections","rapidDetections","stratDetections"],
    ["survDetRate","patDetRate","rapidDetRate","stratDetRate"],
    ["survRMSE","patRMSE","rapidRMSE","stratRMSE"],
    ["survCoverage","patCoverage","rapidCoverage","stratCoverage"],
    ["survSensingDuty","patSensingDuty","rapidSensingDuty","stratSensingDuty"],
    ["survEnergyPerDet","patEnergyPerDet","rapidEnergyPerDet","stratEnergyPerDet"],
):
    epd_val = top20_3d[epd].replace([np.inf,-np.inf],np.nan).dropna().mean()
    role_data.append({
        "Role": role,
        "Avg Detections": f"{top20_3d[det].mean():.0f}",
        "Det Rate (det/s)": f"{top20_3d[rate].mean():.4f}",
        "RMSE (m)": f"{top20_3d[rmse].mean():.2f}",
        "Coverage (%)": f"{top20_3d[cov].mean()*100:.1f}",
        "Duty Cycle (%)": f"{top20_3d[duty].mean()*100:.1f}",
        "Energy/Det (kJ)": f"{epd_val/1000:.3f}",
    })
t8 = pd.DataFrame(role_data)
styled_table(t8,
    "Table T8 — Per-Role UAV Performance Summary (3D, Top 20 Runs)",
    "T8_per_role_summary_3d.png")

# ── TABLE T9: xApp KPI summary ──────────────────────────────
t9 = pd.DataFrame([
    ["t=20s (start)",
     f"{xkpi.iloc[0]['pdet']:.4f}",
     f"{xkpi.iloc[0]['rmse_m']:.2f}",
     f"{xkpi.iloc[0]['jscore']:.4f}",
     f"{xkpi.iloc[0]['energy_j']/1000:.1f}",
     f"{xkpi.iloc[0]['avg_throughput_mbps']*1000:.4f}",
     f"{xkpi.iloc[0]['avg_delay_ms']:.2f}"],
    ["t=150s (mid)",
     f"{xkpi.iloc[len(xkpi)//2]['pdet']:.4f}",
     f"{xkpi.iloc[len(xkpi)//2]['rmse_m']:.2f}",
     f"{xkpi.iloc[len(xkpi)//2]['jscore']:.4f}",
     f"{xkpi.iloc[len(xkpi)//2]['energy_j']/1000:.1f}",
     f"{xkpi.iloc[len(xkpi)//2]['avg_throughput_mbps']*1000:.4f}",
     f"{xkpi.iloc[len(xkpi)//2]['avg_delay_ms']:.2f}"],
    ["t=290s (end)",
     f"{xkpi.iloc[-1]['pdet']:.4f}",
     f"{xkpi.iloc[-1]['rmse_m']:.2f}",
     f"{xkpi.iloc[-1]['jscore']:.4f}",
     f"{xkpi.iloc[-1]['energy_j']/1000:.1f}",
     f"{xkpi.iloc[-1]['avg_throughput_mbps']*1000:.4f}",
     f"{xkpi.iloc[-1]['avg_delay_ms']:.2f}"],
    ["Mean",
     f"{xkpi['pdet'].mean():.4f}",
     f"{xkpi['rmse_m'].mean():.2f}",
     f"{xkpi['jscore'].mean():.4f}",
     "—",
     f"{xkpi['avg_throughput_mbps'].mean()*1000:.4f}",
     f"{xkpi['avg_delay_ms'].mean():.2f}"],
], columns=["Time","Pdet","RMSE (m)","Jscore",
            "Energy (kJ)","Thr (kbps)","Delay (ms)"])
styled_table(t9,
    "Table T9 — xApp KPI Snapshots During Simulation",
    "T9_xapp_kpi_snapshots.png")

# ============================================================
# ██████████████  FIGURES — 2D  ██████████████████████████████
# ============================================================
print("\n── Generating 2D figures ──")

# ── F01: Optimisation progress (2D) ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df2d["rngRun"], df2d["Jfinal"], "o-", ms=4, lw=1.2,
        color=C[0], alpha=0.8, label="J-score")
ax.axvline(50, color="red", ls="--", lw=1.5,
           label="RL optimisation begins")
ax.axhline(base2d["Jfinal"].mean(), color="grey", ls=":", lw=1,
           label=f"Baseline mean ({base2d['Jfinal'].mean():.2f})")
ax.set_xlabel("Run number"); ax.set_ylabel("J-score")
ax.set_title("Optimisation Progress Across Simulation Runs (2D)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); save("F01_optim_progress_2d.png")

# ── F02: RMSE vs run (2D) ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df2d["rngRun"], df2d["RMSEm"], "o-", ms=4, lw=1.2,
        color=C[2], alpha=0.8)
ax.axvline(50, color="red", ls="--", lw=1.5,
           label="RL optimisation begins")
ax.set_xlabel("Run number"); ax.set_ylabel("RMSE (m)")
ax.set_title("Localisation RMSE Across Simulation Runs (2D)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); save("F02_rmse_vs_run_2d.png")

# ── F03: RMSE boxplot before/after (2D) ─────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.boxplot([base2d["RMSEm"], post2d["RMSEm"]],
    labels=["Exploration\n(runs 1–50)", "RL-Guided\n(runs 51+)"],
    patch_artist=True,
    boxprops=dict(facecolor="#aec7e8"),
    medianprops=dict(color="red", lw=2))
ax.set_ylabel("RMSE (m)")
ax.set_title("Localisation RMSE: Exploration vs RL-Guided (2D)")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout(); save("F03_rmse_boxplot_2d.png")

# ── F04: Pdet vs run (2D) ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df2d["rngRun"], df2d["Pdet"], "o-", ms=4, lw=1.2,
        color=C[3], alpha=0.8)
ax.axvline(50, color="red", ls="--", lw=1.5,
           label="RL optimisation begins")
ax.axhline(0.97, color="grey", ls=":", lw=1, label="Target (0.97)")
ax.set_ylim(0, 1.05)
ax.set_xlabel("Run number"); ax.set_ylabel("Detection probability")
ax.set_title("Detection Probability Across Simulation Runs (2D)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); save("F04_pdet_vs_run_2d.png")

# ── F05: Pdet distribution (2D) ─────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(base2d["Pdet"], bins=15, alpha=0.7,
        label="Exploration (1–50)", color=C[0])
ax.hist(post2d["Pdet"], bins=15, alpha=0.7,
        label="RL-Guided (51+)", color=C[1])
ax.set_xlabel("Detection probability"); ax.set_ylabel("Count")
ax.set_title("Detection Probability Distribution by Phase (2D)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); save("F05_pdet_distribution_2d.png")

# ── F06: Energy vs RMSE tradeoff (2D) ───────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(df2d["EtotJ"]/1000, df2d["RMSEm"],
                c=df2d["Jfinal"], cmap="viridis", s=40, alpha=0.8)
plt.colorbar(sc, ax=ax, label="J-score")
ax.set_xlabel("Total energy (kJ)"); ax.set_ylabel("RMSE (m)")
ax.set_title("Energy–Accuracy Trade-off (2D)")
ax.grid(True, alpha=0.3)
plt.tight_layout(); save("F06_energy_rmse_tradeoff_2d.png")

# ── F07: Jscore vs Energy (2D) ──────────────────────────────
colors_phase_2d = ["#1f77b4" if r <= 50 else "#ff7f0e"
                   for r in df2d["rngRun"]]
legend_els = [Patch(fc="#1f77b4", label="Exploration (1–50)"),
              Patch(fc="#ff7f0e", label="RL-Guided (51+)")]
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df2d["EtotJ"]/1000, df2d["Jfinal"],
           c=colors_phase_2d, s=40, alpha=0.8)
ax.legend(handles=legend_els, fontsize=9)
ax.set_xlabel("Total energy (kJ)"); ax.set_ylabel("J-score")
ax.set_title("J-Score vs Energy by Phase (2D)")
ax.grid(True, alpha=0.3)
plt.tight_layout(); save("F07_jscore_vs_energy_2d.png")

# ── F08: Jscore vs RMSE (2D) ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df2d["RMSEm"], df2d["Jfinal"],
           c=colors_phase_2d, s=40, alpha=0.8)
ax.legend(handles=legend_els, fontsize=9)
ax.set_xlabel("RMSE (m)"); ax.set_ylabel("J-score")
ax.set_title("J-Score vs RMSE by Phase (2D)")
ax.grid(True, alpha=0.3)
plt.tight_layout(); save("F08_jscore_vs_rmse_2d.png")

# ── F09: Jscore vs Pdet (2D) ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df2d["Pdet"], df2d["Jfinal"],
           c=colors_phase_2d, s=40, alpha=0.8)
ax.legend(handles=legend_els, fontsize=9)
ax.set_xlabel("Pdet"); ax.set_ylabel("J-score")
ax.set_title("J-Score vs Detection Probability by Phase (2D)")
ax.grid(True, alpha=0.3)
plt.tight_layout(); save("F09_jscore_vs_pdet_2d.png")

# ── F10: Feature importance ─────────────────────────────────
if feat is not None:
    fc = feat.columns[-1]; nc = feat.columns[0]
    ft10 = feat.sort_values(fc, ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(ft10[nc][::-1], ft10[fc][::-1], color=C[0])
    ax.set_xlabel("Importance score")
    ax.set_title("Top 10 Most Influential Optimisation Parameters")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout(); save("F10_feature_importance.png")

# ── F11: Per-UAV energy (2D top 10) ─────────────────────────
uav_cols = ["UAV0_Energy_J","UAV1_Energy_J",
            "UAV2_Energy_J","UAV3_Energy_J"]
uav_means_2d = [top20_2d[c].mean()/1000 for c in uav_cols]
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(ROLES, uav_means_2d, color=C)
ax.set_ylabel("Mean energy (kJ)")
ax.set_title("Per-UAV Energy Consumption (Top 20 Runs, 2D)")
for bar, val in zip(bars, uav_means_2d):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.3, f"{val:.1f}",
            ha="center", va="bottom", fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout(); save("F11_per_uav_energy_2d.png")

# ── F12: Energy breakdown (2D) ──────────────────────────────
ep = df2d["EpropJ"].mean()/1000
er = df2d["ERFJ"].mean()/1000
ec = df2d["EprocJ"].mean()/1000
total = ep + er + ec
fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(["Propulsion","RF Tx","Processing"], [ep, er, ec], color=C[:3])
ax.set_ylabel("Mean energy (kJ)")
ax.set_title("Mean Energy Breakdown (2D, All Runs)")
for i, val in enumerate([ep, er, ec]):
    ax.text(i, val+0.1, f"{val/total*100:.1f}%",
            ha="center", va="bottom", fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout(); save("F12_energy_breakdown_2d.png")

# ── F13: Communication KPIs vs run (2D) ─────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col, lbl, c, unit in zip(
    axes,
    ["avgThrMbps","avgDelayMs","avgLossPct"],
    ["Throughput","Delay","Packet Loss"],
    C, ["kbps","ms","%"]
):
    vals = df2d[col]*1000 if "Thr" in lbl else df2d[col]
    ax.plot(df2d["rngRun"], vals, "o-", ms=3, lw=1.2, color=c)
    ax.axvline(50, color="red", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("Run number")
    ax.set_ylabel(f"{lbl} ({unit})")
    ax.set_title(f"{lbl} vs Run (2D)")
    ax.grid(True, alpha=0.3)
plt.suptitle("Communication KPIs Across Runs (2D)",
             fontsize=10, fontweight="bold")
plt.tight_layout(); save("F13_comms_kpis_2d.png")

# ── F14: Sensing interval vs Jscore per role (2D) ───────────
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, col, lbl, c in zip(
    axes,
    ["survSenseDt","patSenseDt","rapidSenseDt","stratSenseDt"],
    ROLES, C
):
    ax.scatter(df2d[col], df2d["Jfinal"], color=c, s=25, alpha=0.7)
    ax.set_xlabel(f"{lbl}\nsensing interval (s)", fontsize=8)
    ax.set_ylabel("J-score", fontsize=8)
    ax.grid(True, alpha=0.3)
plt.suptitle("Sensing Interval vs J-Score per UAV Role (2D)",
             fontsize=10, fontweight="bold")
plt.tight_layout(); save("F14_sensing_interval_vs_jscore_2d.png")

# ── F15: Alpha vs Jscore per role (2D) ──────────────────────
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, col, lbl, c in zip(
    axes,
    ["alphaSurv","alphaPat","alphaRapid","alphaStrat"],
    ROLES, C
):
    ax.scatter(df2d[col], df2d["Jfinal"], color=c, s=25, alpha=0.7)
    ax.set_xlabel(f"{lbl}\nα", fontsize=8)
    ax.set_ylabel("J-score", fontsize=8)
    ax.grid(True, alpha=0.3)
plt.suptitle("Gauss-Markov α vs J-Score per UAV Role (2D)",
             fontsize=10, fontweight="bold")
plt.tight_layout(); save("F15_alpha_vs_jscore_2d.png")

# ── F16: Vmax vs Jscore per role (2D) ───────────────────────
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, col, lbl, c in zip(
    axes,
    ["survVmax","patVmax","rapidVmax","stratVmax"],
    ROLES, C
):
    ax.scatter(df2d[col], df2d["Jfinal"], color=c, s=25, alpha=0.7)
    ax.set_xlabel(f"{lbl}\nVmax (m/s)", fontsize=8)
    ax.set_ylabel("J-score", fontsize=8)
    ax.grid(True, alpha=0.3)
plt.suptitle("Max Speed vs J-Score per UAV Role (2D)",
             fontsize=10, fontweight="bold")
plt.tight_layout(); save("F16_vmax_vs_jscore_2d.png")

# ── F17: Altitude vs Jscore per role (2D) ───────────────────
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, col, lbl, c in zip(
    axes,
    ["survZmax","patZmax","rapidZmax","stratZmax"],
    ROLES, C
):
    ax.scatter(df2d[col], df2d["Jfinal"], color=c, s=25, alpha=0.7)
    ax.set_xlabel(f"{lbl}\nZmax (m)", fontsize=8)
    ax.set_ylabel("J-score", fontsize=8)
    ax.grid(True, alpha=0.3)
plt.suptitle("Max Altitude vs J-Score per UAV Role (2D)",
             fontsize=10, fontweight="bold")
plt.tight_layout(); save("F17_altitude_vs_jscore_2d.png")

# ── F18: Correlation heatmap (2D) ───────────────────────────
corr_cols = ["Pdet","RMSEm","Jfinal","EtotJ",
             "avgThrMbps","avgDelayMs","avgLossPct"]

# Drop columns with zero or near-zero variance (causes NaN in corr)
corr_data = df2d[corr_cols].copy()
corr_data = corr_data.loc[:, corr_data.std() > 1e-6]
corr = corr_data.corr()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, label="Correlation")
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(corr.columns, fontsize=9)
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        val = corr.iloc[i,j]
        txt = f"{val:.2f}" if not np.isnan(val) else "—"
        ax.text(j, i, txt,
                ha="center", va="center", fontsize=8)
ax.set_title("Performance Metric Correlation Matrix (2D)")
plt.tight_layout(); save("F18_correlation_heatmap_2d.png")

# ============================================================
# ██████████████  FIGURES — 3D  ██████████████████████████████
# ============================================================
print("\n── Generating 3D figures ──")

colors_phase_3d = ["#1f77b4" if r <= 50 else "#ff7f0e"
                   for r in df3d["rngRun"]]

# ── G01: Optimisation progress (3D) ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df3d["rngRun"], df3d["Jfinal"], "o-", ms=4, lw=1.2,
        color=C[0], alpha=0.8)
ax.axvline(50, color="red", ls="--", lw=1.5,
           label="RL optimisation begins")
ax.axhline(base3d["Jfinal"].mean(), color="grey", ls=":", lw=1,
           label=f"Baseline mean ({base3d['Jfinal'].mean():.2f})")
ax.set_xlabel("Run number"); ax.set_ylabel("J-score")
ax.set_title("Optimisation Progress Across Simulation Runs (3D)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); save("G01_optim_progress_3d.png")

# ── G02: RMSE vs run (3D) ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df3d["rngRun"], df3d["RMSEm_3D"], "o-", ms=4, lw=1.2,
        color=C[2], alpha=0.8)
ax.axvline(50, color="red", ls="--", lw=1.5,
           label="RL optimisation begins")
ax.set_xlabel("Run number"); ax.set_ylabel("RMSE 3D (m)")
ax.set_title("Localisation RMSE Across Simulation Runs (3D)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); save("G02_rmse_vs_run_3d.png")

# ── G03: RMSE boxplot before/after (3D) ─────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.boxplot([base3d["RMSEm_3D"], post3d["RMSEm_3D"]],
    labels=["Exploration\n(runs 1–50)", "RL-Guided\n(runs 51+)"],
    patch_artist=True,
    boxprops=dict(facecolor="#aec7e8"),
    medianprops=dict(color="red", lw=2))
ax.set_ylabel("RMSE 3D (m)")
ax.set_title("Localisation RMSE: Exploration vs RL-Guided (3D)")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout(); save("G03_rmse_boxplot_3d.png")

# ── G04: Pdet vs run (3D) ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df3d["rngRun"], df3d["Pdet"], "o-", ms=4, lw=1.2,
        color=C[3], alpha=0.8)
ax.axvline(50, color="red", ls="--", lw=1.5,
           label="RL optimisation begins")
ax.axhline(0.97, color="grey", ls=":", lw=1, label="Target (0.97)")
ax.set_ylim(0, 1.05)
ax.set_xlabel("Run number"); ax.set_ylabel("Detection probability")
ax.set_title("Detection Probability Across Simulation Runs (3D)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); save("G04_pdet_vs_run_3d.png")

# ── G05: Pdet distribution (3D) ─────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(base3d["Pdet"], bins=15, alpha=0.7,
        label="Exploration", color=C[0])
ax.hist(post3d["Pdet"], bins=15, alpha=0.7,
        label="RL-Guided", color=C[1])
ax.set_xlabel("Detection probability"); ax.set_ylabel("Count")
ax.set_title("Detection Probability Distribution (3D)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); save("G05_pdet_distribution_3d.png")

# ── G06: Energy-RMSE tradeoff (3D) ──────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(df3d["EtotJ"]/1000, df3d["RMSEm_3D"],
                c=df3d["Jfinal"], cmap="viridis", s=40, alpha=0.8)
plt.colorbar(sc, ax=ax, label="J-score")
ax.set_xlabel("Total energy (kJ)"); ax.set_ylabel("RMSE 3D (m)")
ax.set_title("Energy–Accuracy Trade-off (3D)")
ax.grid(True, alpha=0.3)
plt.tight_layout(); save("G06_energy_rmse_tradeoff_3d.png")

# ── G07: Jscore vs Energy (3D) ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df3d["EtotJ"]/1000, df3d["Jfinal"],
           c=colors_phase_3d, s=40, alpha=0.8)
ax.legend(handles=legend_els, fontsize=9)
ax.set_xlabel("Total energy (kJ)"); ax.set_ylabel("J-score")
ax.set_title("J-Score vs Energy by Phase (3D)")
ax.grid(True, alpha=0.3)
plt.tight_layout(); save("G07_jscore_vs_energy_3d.png")

# ── G08: Jscore vs RMSE (3D) ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df3d["RMSEm_3D"], df3d["Jfinal"],
           c=colors_phase_3d, s=40, alpha=0.8)
ax.legend(handles=legend_els, fontsize=9)
ax.set_xlabel("RMSE 3D (m)"); ax.set_ylabel("J-score")
ax.set_title("J-Score vs RMSE by Phase (3D)")
ax.grid(True, alpha=0.3)
plt.tight_layout(); save("G08_jscore_vs_rmse_3d.png")

# ── G09: Jscore vs Pdet (3D) ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df3d["Pdet"], df3d["Jfinal"],
           c=colors_phase_3d, s=40, alpha=0.8)
ax.legend(handles=legend_els, fontsize=9)
ax.set_xlabel("Pdet"); ax.set_ylabel("J-score")
ax.set_title("J-Score vs Detection Probability (3D)")
ax.grid(True, alpha=0.3)
plt.tight_layout(); save("G09_jscore_vs_pdet_3d.png")

# ── G10: Per-role detection rate (3D) ───────────────────────
means_det = [top20_3d[c].mean() for c in
             ["survDetRate","patDetRate","rapidDetRate","stratDetRate"]]
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(ROLES, means_det, color=C)
ax.set_ylabel("Mean detection rate (det/s)")
ax.set_title("Per-Role Detection Rate (Top 20 Runs, 3D)")
for bar, val in zip(bars, means_det):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.001, f"{val:.4f}",
            ha="center", va="bottom", fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout(); save("G10_per_role_detection_rate_3d.png")

# ── G11: Per-role RMSE (3D) ─────────────────────────────────
means_rmse = [top20_3d[c].mean() for c in
              ["survRMSE","patRMSE","rapidRMSE","stratRMSE"]]
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(ROLES, means_rmse, color=C)
ax.set_ylabel("Mean RMSE (m)")
ax.set_title("Per-Role Localisation RMSE (Top 20 Runs, 3D)")
for bar, val in zip(bars, means_rmse):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.2, f"{val:.1f}m",
            ha="center", va="bottom", fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout(); save("G11_per_role_rmse_3d.png")

# ── G12: Per-role coverage (3D) ─────────────────────────────
means_cov = [top20_3d[c].mean()*100 for c in
             ["survCoverage","patCoverage",
              "rapidCoverage","stratCoverage"]]
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(ROLES, means_cov, color=C)
ax.set_ylabel("Mean area coverage (%)")
ax.set_title("Per-Role Spatial Coverage (Top 20 Runs, 3D)")
for bar, val in zip(bars, means_cov):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.3, f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout(); save("G12_per_role_coverage_3d.png")

# ── G13: Per-role energy per detection (3D) ─────────────────
epd_vals = [
    top20_3d[c].replace([np.inf,-np.inf],np.nan).dropna().mean()/1000
    for c in ["survEnergyPerDet","patEnergyPerDet",
              "rapidEnergyPerDet","stratEnergyPerDet"]
]
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(ROLES, epd_vals, color=C)
ax.set_ylabel("Energy per detection (kJ/det)")
ax.set_title("Per-Role Energy Efficiency (Top 20 Runs, 3D)")
for bar, val in zip(bars, epd_vals):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.01, f"{val:.2f}",
            ha="center", va="bottom", fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout(); save("G13_per_role_energy_per_det_3d.png")

# ── G14: Per-role sensing duty cycle (3D) ───────────────────
duty_vals = [top20_3d[c].mean()*100 for c in
             ["survSensingDuty","patSensingDuty",
              "rapidSensingDuty","stratSensingDuty"]]
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(ROLES, duty_vals, color=C)
ax.set_ylabel("Sensing duty cycle (%)")
ax.set_title("Per-Role Sensing Duty Cycle (Top 20 Runs, 3D)")
for bar, val in zip(bars, duty_vals):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.2, f"{val:.1f}%",
            ha="center", va="bottom", fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout(); save("G14_per_role_sensing_duty_3d.png")

# ── G15: Per-UAV energy (3D) ────────────────────────────────
uav_means_3d = [top20_3d[c].mean()/1000 for c in uav_cols]
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(ROLES, uav_means_3d, color=C)
ax.set_ylabel("Mean energy (kJ)")
ax.set_title("Per-UAV Energy Consumption (Top 20 Runs, 3D)")
for bar, val in zip(bars, uav_means_3d):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.3, f"{val:.1f}",
            ha="center", va="bottom", fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout(); save("G15_per_uav_energy_3d.png")

# ── G16: Energy breakdown (3D) ──────────────────────────────
ep3 = df3d["EpropJ"].mean()/1000
er3 = df3d["ERFJ"].mean()/1000
ec3 = df3d["EprocJ"].mean()/1000
tot3 = ep3+er3+ec3
fig, ax = plt.subplots(figsize=(6, 5))
ax.bar(["Propulsion","RF Tx","Processing"], [ep3,er3,ec3], color=C[:3])
ax.set_ylabel("Mean energy (kJ)")
ax.set_title("Mean Energy Breakdown (3D, All Runs)")
for i, val in enumerate([ep3,er3,ec3]):
    ax.text(i, val+0.1, f"{val/tot3*100:.1f}%",
            ha="center", va="bottom", fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout(); save("G16_energy_breakdown_3d.png")

# ── G17: Communication KPIs (3D) ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col, lbl, c, unit, scale in zip(
    axes,
    ["avgThrMbps","avgDelayMs","avgLossPct"],
    ["Throughput","Delay","Packet Loss"],
    C, ["kbps","ms","%"], [1000,1,1]
):
    ax.plot(df3d["rngRun"], df3d[col]*scale,
            "o-", ms=3, lw=1.2, color=c)
    ax.axvline(50, color="red", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("Run number")
    ax.set_ylabel(f"{lbl} ({unit})")
    ax.set_title(f"{lbl} vs Run (3D)")
    ax.grid(True, alpha=0.3)
plt.suptitle("Communication KPIs Across Runs (3D)",
             fontsize=10, fontweight="bold")
plt.tight_layout(); save("G17_comms_kpis_3d.png")

# ── G18: Correlation heatmap (3D) ───────────────────────────
corr3_cols = ["Pdet","RMSEm_3D","Jfinal","EtotJ",
              "avgThrMbps","avgDelayMs","avgLossPct"]

corr3_data = df3d[corr3_cols].copy()
corr3_data = corr3_data.loc[:, corr3_data.std() > 1e-6]
corr3 = corr3_data.corr()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr3, cmap="RdYlGn", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, label="Correlation")
ax.set_xticks(range(len(corr3.columns)))
ax.set_yticks(range(len(corr3.columns)))
ax.set_xticklabels(corr3.columns, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(corr3.columns, fontsize=9)
for i in range(len(corr3.columns)):
    for j in range(len(corr3.columns)):
        val = corr3.iloc[i,j]
        txt = f"{val:.2f}" if not np.isnan(val) else "—"
        ax.text(j, i, txt,
                ha="center", va="center", fontsize=8)
ax.set_title("Performance Metric Correlation Matrix (3D)")
plt.tight_layout(); save("G18_correlation_heatmap_3d.png")

# ── G19: Per-role detection rate over sim runs (3D) ─────────
fig, ax = plt.subplots(figsize=(10, 5))
for col, lbl, c in zip(
    ["survDetRate","patDetRate","rapidDetRate","stratDetRate"],
    ROLES, C
):
    ax.plot(df3d["rngRun"], df3d[col], "o-", ms=3, lw=1.2,
            label=lbl, color=c, alpha=0.8)
ax.axvline(50, color="red", ls="--", lw=1.5,
           label="RL optimisation begins")
ax.set_xlabel("Run number")
ax.set_ylabel("Detection rate (det/s)")
ax.set_title("Per-Role Detection Rate Across Runs (3D)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout(); save("G19_per_role_det_rate_vs_run_3d.png")

# ── G20: Per-role RMSE across runs (3D) ─────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for col, lbl, c in zip(
    ["survRMSE","patRMSE","rapidRMSE","stratRMSE"],
    ROLES, C
):
    ax.plot(df3d["rngRun"], df3d[col], "o-", ms=3, lw=1.2,
            label=lbl, color=c, alpha=0.8)
ax.axvline(50, color="red", ls="--", lw=1.5,
           label="RL optimisation begins")
ax.set_xlabel("Run number")
ax.set_ylabel("RMSE (m)")
ax.set_title("Per-Role Localisation RMSE Across Runs (3D)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout(); save("G20_per_role_rmse_vs_run_3d.png")

# ============================================================
# ██████████████  FIGURES — xApp  ████████████████████████████
# ============================================================
print("\n── Generating xApp figures ──")

# ── X01: Pdet over time ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, xkpi["pdet"], "b-o", ms=5, lw=2)
ax.axhline(0.97, color="red", ls="--", lw=1.5, label="Target (0.97)")
ax.fill_between(t, xkpi["pdet"], 0.97,
                where=xkpi["pdet"] < 0.97,
                alpha=0.15, color="red", label="Below target")
ax.set_ylim(0, 1.05)
ax.set_xlabel("Simulation time (s)")
ax.set_ylabel("Detection probability")
ax.set_title("xApp Control: Detection Probability Over Time")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); save("X01_xapp_pdet.png")

# ── X02: RMSE over time ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, xkpi["rmse_m"], "g-o", ms=5, lw=2)
ax.axhline(25.0, color="red", ls="--", lw=1.5, label="Target (25m)")
ax.set_xlabel("Simulation time (s)")
ax.set_ylabel("RMSE (m)")
ax.set_title("xApp Control: Localisation RMSE Over Time")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); save("X02_xapp_rmse.png")

# ── X03: Jscore over time ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, xkpi["jscore"], "purple", marker="o", ms=5, lw=2,
        label="xApp J-score")
ax.axhline(best_off_2d["Jfinal"], color="orange", ls="--", lw=1.5,
           label=f"Best offline ({best_off_2d['Jfinal']:.3f})")
ax.axhline(base2d["Jfinal"].mean(), color="grey", ls=":", lw=1.2,
           label=f"Baseline mean ({base2d['Jfinal'].mean():.2f})")
ax.set_xlabel("Simulation time (s)")
ax.set_ylabel("J-score")
ax.set_title("xApp Control: J-Score Over Time")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); save("X03_xapp_jscore.png")

# ── X04: Sensing intervals over time ────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for col, lbl, c in zip(
    ["survSenseDt","patSenseDt","rapidSenseDt","stratSenseDt"],
    ROLES, C
):
    ax.plot(t, xkpi[col], marker="o", ms=3, lw=1.5,
            label=lbl, color=c)
ax.set_xlabel("Simulation time (s)")
ax.set_ylabel("Sensing interval (s)")
ax.set_title("xApp Control: Sensing Interval Adaptation")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout(); save("X04_xapp_sensing_intervals.png")

# ── X05: Speed evolution ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for col, lbl, c in zip(
    ["survVmax","patVmax","rapidVmax","stratVmax"],
    ROLES, C
):
    ax.plot(t, xkpi[col], marker="o", ms=3, lw=1.5,
            label=lbl, color=c)
ax.set_xlabel("Simulation time (s)")
ax.set_ylabel("Max speed (m/s)")
ax.set_title("xApp Control: Maximum Speed Adaptation")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout(); save("X05_xapp_speed.png")

# ── X06: Altitude evolution ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for col, lbl, c in zip(
    ["survZmax","patZmax","rapidZmax","stratZmax"],
    ROLES, C
):
    ax.plot(t, xkpi[col], marker="o", ms=3, lw=1.5,
            label=lbl, color=c)
ax.set_xlabel("Simulation time (s)")
ax.set_ylabel("Max altitude (m)")
ax.set_title("xApp Control: Altitude Bound Adaptation")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout(); save("X06_xapp_altitude.png")

# ── X07: Per-UAV energy over time ───────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for i, (lbl, c) in enumerate(zip(ROLES, C)):
    ax.plot(t, xkpi[f"uav{i}_energy_j"]/1000,
            marker="o", ms=3, lw=1.5, label=lbl, color=c)
ax.set_xlabel("Simulation time (s)")
ax.set_ylabel("Cumulative energy (kJ)")
ax.set_title("Per-UAV Energy Accumulation Under xApp Control")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout(); save("X07_xapp_per_uav_energy.png")

# ── X08: Energy breakdown stacked bar (final) ───────────────
final = xkpi.iloc[-1]
p_v = [final[f"uav{i}_eprop_j"]/1000 for i in range(4)]
r_v = [final[f"uav{i}_erf_j"]/1000   for i in range(4)]
c_v = [final[f"uav{i}_eproc_j"]/1000 for i in range(4)]
x_ax = np.arange(4)
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x_ax, p_v, label="Propulsion", color=C[0])
ax.bar(x_ax, r_v, bottom=p_v, label="RF Tx", color=C[1])
ax.bar(x_ax, c_v,
       bottom=[p+r for p,r in zip(p_v,r_v)],
       label="Processing", color=C[2])
ax.set_xticks(x_ax); ax.set_xticklabels(ROLES, fontsize=9)
ax.set_ylabel("Energy (kJ)")
ax.set_title("Per-UAV Energy Breakdown at End of xApp Run")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout(); save("X08_xapp_energy_breakdown.png")

# ── X09: Throughput over time ───────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, xkpi["avg_throughput_mbps"]*1000, "b-o", ms=4, lw=2)
ax.set_xlabel("Simulation time (s)")
ax.set_ylabel("Throughput (kbps)")
ax.set_title("xApp Run: ISAC Throughput Over Time")
ax.grid(True, alpha=0.3)
plt.tight_layout(); save("X09_xapp_throughput.png")

# ── X10: Delay over time ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, xkpi["avg_delay_ms"], "r-o", ms=4, lw=2)
ax.axhline(20.0, color="grey", ls="--", lw=1.2, label="Target (20ms)")
ax.set_xlabel("Simulation time (s)")
ax.set_ylabel("Delay (ms)")
ax.set_title("xApp Run: End-to-End Delay Over Time")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); save("X10_xapp_delay.png")

# ── X11: Packet loss over time ──────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, xkpi["avg_loss_pct"], "m-o", ms=4, lw=2)
ax.set_xlabel("Simulation time (s)")
ax.set_ylabel("Packet loss (%)")
ax.set_title("xApp Run: Packet Loss Over Time")
ax.grid(True, alpha=0.3)
plt.tight_layout(); save("X11_xapp_packet_loss.png")

# ── X12: Total system energy over time ──────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, xkpi["energy_j"]/1000, "orange", marker="o", ms=4, lw=2)
ax.axhline(350.0, color="red", ls="--", lw=1.5, label="Limit (350kJ)")
ax.set_xlabel("Simulation time (s)")
ax.set_ylabel("Total system energy (kJ)")
ax.set_title("xApp Run: Total System Energy Accumulation")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); save("X12_xapp_total_energy.png")

# ── X13: Mobility timestep evolution ────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for col, lbl, c in zip(
    ["survDt","patDt","rapidDt","stratDt"], ROLES, C
):
    ax.plot(t, xkpi[col], marker="o", ms=3, lw=1.5,
            label=lbl, color=c)
ax.set_xlabel("Simulation time (s)")
ax.set_ylabel("Mobility update interval (s)")
ax.set_title("xApp Control: Mobility Timestep Adaptation")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
plt.tight_layout(); save("X13_xapp_mobility_dt.png")

# ── X14: Parameter convergence summary grid ─────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
param_groups = [
    (["survSenseDt","patSenseDt","rapidSenseDt","stratSenseDt"],
     "Sensing Intervals (s)"),
    (["survVmax","patVmax","rapidVmax","stratVmax"],
     "Max Speed (m/s)"),
    (["survZmax","patZmax","rapidZmax","stratZmax"],
     "Max Altitude (m)"),
    (["survDt","patDt","rapidDt","stratDt"],
     "Mobility Timestep (s)"),
]
for ax, (cols, ylabel) in zip(axes.flat, param_groups):
    for col, lbl, c in zip(cols, ROLES, C):
        ax.plot(t, xkpi[col], marker="o", ms=2, lw=1.2,
                label=lbl, color=c, alpha=0.8)
    ax.set_xlabel("Simulation time (s)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
plt.suptitle("xApp Parameter Convergence Over Simulation Time",
             fontsize=11, fontweight="bold")
plt.tight_layout(); save("X14_xapp_parameter_convergence.png")

# ============================================================
# ██████████████  CENTREPIECE  ███████████████████████████████
# ============================================================
print("\n── Generating centrepiece figures ──")

# ── Z01: Three-way comparison ───────────────────────────────
configs = [
    "Baseline mean\n(runs 1–50)",
    f"Best offline\n(run {int(best_off_2d['rngRun'])})",
    "xApp \n(near-RT RIC)"
]
pdet_v   = [base2d["Pdet"].mean(),   best_off_2d["Pdet"],   xapp_pdet]
rmse_v   = [base2d["RMSEm"].mean(),  best_off_2d["RMSEm"],  xapp_rmse]
jscore_v = [base2d["Jfinal"].mean(), best_off_2d["Jfinal"], xapp_jscore]
energy_v = [base2d["EtotJ"].mean()/1000,
            best_off_2d["EtotJ"]/1000,
            xapp_energy/1000]

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
for ax, vals, title, ylabel in zip(
    axes,
    [pdet_v, rmse_v, jscore_v, energy_v],
    ["Detection Probability",
     "Localisation RMSE (m)",
     "Composite J-Score",
     "Total Energy (kJ)"],
    ["Pdet", "RMSE (m)", "J-score", "Energy (kJ)"]
):
    bars = ax.bar(configs, vals, color=C[:3], width=0.5)
    ax.set_title(title, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xticklabels(configs, fontsize=7)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()*1.01,
                f"{val:.3f}" if val < 10 else f"{val:.1f}",
                ha="center", va="bottom", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
plt.suptitle(
    "System Performance: Baseline vs Offline RL-Optimised vs xApp",
    fontsize=11, fontweight="bold")
plt.tight_layout(); save("Z01_three_way_comparison.png")

# ── Z02: 2D vs 3D optimisation progress overlay ─────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df2d["rngRun"], df2d["Jfinal"], "o-", ms=3, lw=1.2,
        color=C[0], alpha=0.7, label="2D")
ax.plot(df3d["rngRun"], df3d["Jfinal"], "s--", ms=3, lw=1.2,
        color=C[1], alpha=0.7, label="3D")
ax.axvline(50, color="red", ls="--", lw=1.5,
           label="RL optimisation begins")
ax.set_xlabel("Run number"); ax.set_ylabel("J-score")
ax.set_title("Optimisation Progress: 2D vs 3D Dataset Comparison")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); save("Z02_2d_vs_3d_progress.png")

# ── Summary ──────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Total files saved: {len(saved)}")
print(f"Output directory : {OUT}")
print(f"\nTABLES (T01–T09):")
for s in saved:
    if s.startswith("T"):
        desc = {
            "T1":"Before/After Optimisation (2D)",
            "T2":"Before/After Optimisation (3D)",
            "T3":"Localisation Start vs End (2D)",
            "T4":"Localisation Start vs End (3D)",
            "T5":"Improvement vs Baseline (2D)",
            "T6":"Improvement vs Baseline (3D)",
            "T7":"Three-Way Summary",
            "T8":"Per-Role Summary (3D)",
            "T9":"xApp KPI Snapshots",
        }.get(s[:2],"")
        print(f"  {s:<45} {desc}")
print(f"\nFIGURES — 2D (F01–F18):")
print(f"  F01 Optimisation progress | F02 RMSE vs run | F03 RMSE boxplot")
print(f"  F04 Pdet vs run | F05 Pdet dist | F06 Energy-RMSE tradeoff")
print(f"  F07 Jscore vs Energy | F08 Jscore vs RMSE | F09 Jscore vs Pdet")
print(f"  F10 Feature importance | F11 Per-UAV energy | F12 Energy breakdown")
print(f"  F13 Comms KPIs | F14 Sensing interval vs J | F15 Alpha vs J")
print(f"  F16 Vmax vs J | F17 Altitude vs J | F18 Correlation heatmap")
print(f"\nFIGURES — 3D (G01–G20): Same as 2D + per-role plots")
print(f"\nFIGURES — xApp (X01–X14):")
print(f"  X01 Pdet | X02 RMSE | X03 Jscore | X04 Sensing intervals")
print(f"  X05 Speed | X06 Altitude | X07 Per-UAV energy | X08 Energy breakdown")
print(f"  X09 Throughput | X10 Delay | X11 Loss | X12 Total energy")
print(f"  X13 Mobility dt | X14 Parameter convergence")
print(f"\nCENTREPIECE (Z01–Z02):")
print(f"  Z01 Three-way comparison | Z02 2D vs 3D overlay")
print(f"\nDONE")
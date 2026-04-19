import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

CSV_PATH = "run_summary_3d.csv"

FEATURES = [
    "survSenseDt", "patSenseDt", "rapidSenseDt", "stratSenseDt",
    "alphaSurv", "alphaPat", "alphaRapid", "alphaStrat",
    "survVmin", "survVmax", "patVmin", "patVmax",
    "rapidVmin", "rapidVmax", "stratVmin", "stratVmax",
    "survZmin", "survZmax", "patZmin", "patZmax",
    "rapidZmin", "rapidZmax", "stratZmin", "stratZmax",
    "survDt", "patDt", "rapidDt", "stratDt"
    
]

TARGET = "Jfinal"

# Per-role KPI columns for analysis
ROLE_KPIS = [
    "survDetections", "patDetections", "rapidDetections", "stratDetections",
    "survDetRate", "patDetRate", "rapidDetRate", "stratDetRate",
    "survRMSE", "patRMSE", "rapidRMSE", "stratRMSE",
    "survEnergyPerDet", "patEnergyPerDet", "rapidEnergyPerDet", "stratEnergyPerDet",
    "survCoverage", "patCoverage", "rapidCoverage", "stratCoverage",
    "survSensingDuty", "patSensingDuty", "rapidSensingDuty", "stratSensingDuty",
    "survDataPerDet", "patDataPerDet", "rapidDataPerDet", "stratDataPerDet"
]

# Search bounds for candidate generation
BOUNDS = {
    "survSenseDt": (0.5, 2.5),
    "patSenseDt": (0.5, 2.5),
    "rapidSenseDt": (0.5, 2.5),
    "stratSenseDt": (0.5, 2.5),

    "alphaSurv": (0.5, 0.95),
    "alphaPat": (0.5, 0.95),
    "alphaRapid": (0.3, 0.8),
    "alphaStrat": (0.5, 0.95),

    "survVmin": (6.0, 12.0),
    "survVmax": (15.0, 25.0),
    "patVmin": (10.0, 18.0),
    "patVmax": (22.0, 35.0),
    "rapidVmin": (18.0, 25.0),
    "rapidVmax": (30.0, 40.0),
    "stratVmin": (8.0, 15.0),
    "stratVmax": (15.0, 25.0),

    "survZmin": (50.0, 70.0),
    "survZmax": (70.0, 90.0),
    "patZmin": (70.0, 90.0),
    "patZmax": (90.0, 110.0),
    "rapidZmin": (90.0, 110.0),
    "rapidZmax": (110.0, 130.0),
    "stratZmin": (110.0, 130.0),
    "stratZmax": (130.0, 150.0),

    "survDt": (0.8, 2.5),
    "patDt": (0.8, 2.5),
    "rapidDt": (0.5, 2.0),
    "stratDt": (0.8, 2.5),

    "terrainHillHeight": (5.0, 30.0),
    "terrainHillFreq": (0.005, 0.02),
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=FEATURES + [TARGET]).copy()
    return df


def train_model(df: pd.DataFrame) -> tuple[RandomForestRegressor, pd.DataFrame, pd.Series]:
    X = df[FEATURES]
    y = df[TARGET]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X, y)
    return model, X, y


def evaluate_model(model: RandomForestRegressor, X: pd.DataFrame, y: pd.Series) -> None:
    pred_train = model.predict(X)
    train_mae = mean_absolute_error(y, pred_train)
    train_r2 = r2_score(y, pred_train)

    loo = LeaveOneOut()
    cv_scores = cross_val_score(
        model, X, y, cv=loo, scoring="neg_mean_absolute_error", n_jobs=None
    )
    loo_mae = -cv_scores.mean()

    print("=== Offline model results (3D) ===")
    print(f"Rows: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Train R2 : {train_r2:.4f}")
    print(f"LOO-CV MAE: {loo_mae:.4f}")


def save_ranked_runs(df: pd.DataFrame) -> None:
    base_cols = ["rngRun", "Jfinal", "Pdet", "RMSEm_3D", "EtotJ"]
    role_cols = [c for c in ROLE_KPIS if c in df.columns]
    cols = base_cols + FEATURES + role_cols
    cols = [c for c in cols if c in df.columns]
    
    ranked = df.sort_values("Jfinal", ascending=False)[cols].copy()
    ranked.to_csv("offline_ranked_runs_3d.csv", index=False)

    print("\n=== Top 5 observed runs by Jfinal (3D) ===")
    print(ranked.head(5).to_string(index=False))


def save_feature_importance(model: RandomForestRegressor) -> None:
    importance = pd.Series(model.feature_importances_, index=FEATURES)
    importance = importance.sort_values(ascending=False)

    importance.to_csv("offline_feature_importance_3d.csv", header=["importance"])

    print("\n=== Most important action variables (3D) ===")
    print(importance.head(10).to_string())


def compute_role_analytics(df: pd.DataFrame) -> None:
    """Compute and display per-role analytics."""
    print("\n=== PER-ROLE ANALYTICS (3D) ===")
    
    roles = ["surv", "pat", "rapid", "strat"]
    role_names = ["Surveillance", "Patrol", "Rapid", "Strategic"]
    
    for role, name in zip(roles, role_names):
        det_col = f"{role}Detections"
        rate_col = f"{role}DetRate"
        rmse_col = f"{role}RMSE"
        energy_col = f"{role}EnergyPerDet"
        coverage_col = f"{role}Coverage"
        duty_col = f"{role}SensingDuty"
        
        print(f"\n{name}:")
        if det_col in df.columns:
            print(f"  Avg Detections: {df[det_col].mean():.1f}")
        if rate_col in df.columns:
            print(f"  Avg Detection Rate: {df[rate_col].mean():.3f} det/s")
        if rmse_col in df.columns:
            print(f"  Avg RMSE 3D: {df[rmse_col].mean():.2f} m")
        if energy_col in df.columns:
            finite_energy = df[energy_col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(finite_energy) > 0:
                print(f"  Avg Energy/Detection: {finite_energy.mean():.1f} J")
        if coverage_col in df.columns:
            print(f"  Avg Coverage: {df[coverage_col].mean() * 100:.1f}%")
        if duty_col in df.columns:
            print(f"  Avg Sensing Duty: {df[duty_col].mean() * 100:.1f}%")


def compute_detection_complementarity(df: pd.DataFrame) -> None:
    """Analyze how roles complement each other in detection coverage."""
    print("\n=== DETECTION COMPLEMENTARITY ===")
    
    det_cols = ["survDetections", "patDetections", "rapidDetections", "stratDetections"]
    available_cols = [c for c in det_cols if c in df.columns]
    
    if len(available_cols) < 2:
        print("Insufficient role detection data for complementarity analysis.")
        return
    
    det_matrix = df[available_cols]
    
    # Correlation between role detections
    corr = det_matrix.corr()
    print("\nDetection correlation matrix:")
    print(corr.to_string())
    
    # Role utilization efficiency: detections / total detections
    total_det = det_matrix.sum(axis=1)
    for col in available_cols:
        role = col.replace("Detections", "")
        util = (df[col] / total_det.replace(0, np.nan)).mean() * 100
        print(f"{role} utilization: {util:.1f}%")


def compute_sensing_comm_tradeoff(df: pd.DataFrame) -> None:
    """Analyze sensing vs communication tradeoff metrics."""
    print("\n=== SENSING-COMMUNICATION TRADEOFF ===")
    
    if "avgThrMbps" in df.columns and "Pdet" in df.columns:
        # Sensing-to-communication ratio approximation
        # Higher Pdet with lower throughput overhead = better ratio
        df_analysis = df.copy()
        df_analysis["sensing_comm_score"] = df_analysis["Pdet"] / (df_analysis["avgThrMbps"] + 0.001)
        
        print(f"Avg Sensing-Comm Score: {df_analysis['sensing_comm_score'].mean():.2f}")
        print(f"Best run (high sensing, low comm overhead):")
        best_idx = df_analysis["sensing_comm_score"].idxmax()
        print(f"  Run {df.loc[best_idx, 'rngRun']}: Pdet={df.loc[best_idx, 'Pdet']:.3f}, Thr={df.loc[best_idx, 'avgThrMbps']:.4f} Mbps")


def sample_candidates(n: int = 20000, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {}

    for feature in FEATURES:
        lo, hi = BOUNDS[feature]
        data[feature] = rng.uniform(lo, hi, size=n)

    cand = pd.DataFrame(data)

    # Enforce structural constraints
    for role in ["surv", "pat", "rapid", "strat"]:
        cand[f"{role}Vmax"] = np.maximum(cand[f"{role}Vmax"], cand[f"{role}Vmin"] + 1.0)
        cand[f"{role}Zmax"] = np.maximum(cand[f"{role}Zmax"], cand[f"{role}Zmin"] + 5.0)

    # Clip to bounds
    for col in cand.columns:
        if col in BOUNDS:
            cand[col] = np.clip(cand[col], BOUNDS[col][0], BOUNDS[col][1])

    return cand


def find_best_candidates(
    model: RandomForestRegressor,
    df_existing: pd.DataFrame,
    top_k: int = 5,
    n_samples: int = 50000,
    seed: int | None = None,
    round_decimals: int = 3,
) -> pd.DataFrame:
    if seed is None:
        seed = int(df_existing["rngRun"].max()) + 1000

    candidates = sample_candidates(n=n_samples, seed=seed)
    candidates["predicted_Jfinal"] = model.predict(candidates[FEATURES])

    existing_keys = set(
        map(tuple, df_existing[FEATURES].round(round_decimals).to_numpy())
    )

    candidate_keys = list(
        map(tuple, candidates[FEATURES].round(round_decimals).to_numpy())
    )
    candidates["config_key"] = candidate_keys

    candidates = candidates[~candidates["config_key"].isin(existing_keys)].copy()
    candidates = candidates.drop_duplicates(subset=["config_key"]).copy()

    best = candidates.sort_values("predicted_Jfinal", ascending=False).head(top_k).copy()
    best = best.drop(columns=["config_key"])
    best.to_csv("offline_suggested_candidates_3d.csv", index=False)

    print("\n=== Top 5 model-suggested NEW candidate configurations (3D) ===")
    show_cols = ["predicted_Jfinal"] + FEATURES
    print(best[show_cols].to_string(index=False))

    return best


def print_ns3_commands(best: pd.DataFrame, start_rng_run: int) -> None:
    print("\n=== ns-3 commands for suggested candidates (3D) ===")
    for i, (_, row) in enumerate(best.iterrows(), start=start_rng_run):
        cmd = [
            './ns3 run "scratch/uav-optimisation-3d',
            f'--RngRun={i}',
        ]
        for feature in FEATURES:
            cmd.append(f'--{feature}={row[feature]:.6f}')
        cmd.append('"')
        print(" ".join(cmd))


def main() -> None:
    df = load_data()
    model, X, y = train_model(df)

    evaluate_model(model, X, y)
    save_ranked_runs(df)
    save_feature_importance(model)
    
    # New analytics
    compute_role_analytics(df)
    compute_detection_complementarity(df)
    compute_sensing_comm_tradeoff(df)

    best = find_best_candidates(model, df_existing=df, top_k=5)

    next_run = int(df["rngRun"].max()) + 1
    print_ns3_commands(best, start_rng_run=next_run)

    print("\nSaved:")
    print(" - offline_ranked_runs_3d.csv")
    print(" - offline_feature_importance_3d.csv")
    print(" - offline_suggested_candidates_3d.csv")


if __name__ == "__main__":
    main()

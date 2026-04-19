import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

CSV_PATH = "run_summary.csv"

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

    print("=== Offline model results ===")
    print(f"Rows: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Train R2 : {train_r2:.4f}")
    print(f"LOO-CV MAE: {loo_mae:.4f}")


def save_ranked_runs(df: pd.DataFrame) -> None:
    cols = ["rngRun", "Jfinal", "Pdet", "RMSEm", "EtotJ"] + FEATURES
    ranked = df.sort_values("Jfinal", ascending=False)[cols].copy()
    ranked.to_csv("offline_ranked_runs.csv", index=False)

    print("\n=== Top 5 observed runs by Jfinal ===")
    print(ranked.head(5).to_string(index=False))


def save_feature_importance(model: RandomForestRegressor) -> None:
    importance = pd.Series(model.feature_importances_, index=FEATURES)
    importance = importance.sort_values(ascending=False)

    importance.to_csv("offline_feature_importance.csv", header=["importance"])

    print("\n=== Most important action variables ===")
    print(importance.head(10).to_string())


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

    # Keep max values within allowed search bounds
    cand["survVmax"] = np.clip(cand["survVmax"], *BOUNDS["survVmax"])
    cand["patVmax"] = np.clip(cand["patVmax"], *BOUNDS["patVmax"])
    cand["rapidVmax"] = np.clip(cand["rapidVmax"], *BOUNDS["rapidVmax"])
    cand["stratVmax"] = np.clip(cand["stratVmax"], *BOUNDS["stratVmax"])

    cand["survZmax"] = np.clip(cand["survZmax"], *BOUNDS["survZmax"])
    cand["patZmax"] = np.clip(cand["patZmax"], *BOUNDS["patZmax"])
    cand["rapidZmax"] = np.clip(cand["rapidZmax"], *BOUNDS["rapidZmax"])
    cand["stratZmax"] = np.clip(cand["stratZmax"], *BOUNDS["stratZmax"])

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

    # Existing run keys as tuples (more robust than string joins)
    existing_keys = set(
        map(tuple, df_existing[FEATURES].round(round_decimals).to_numpy())
    )

    # Candidate keys
    candidate_keys = list(
        map(tuple, candidates[FEATURES].round(round_decimals).to_numpy())
    )
    candidates["config_key"] = candidate_keys

    # Remove already-run configs
    candidates = candidates[~candidates["config_key"].isin(existing_keys)].copy()

    # Remove duplicate candidates within sample set
    candidates = candidates.drop_duplicates(subset=["config_key"]).copy()

    # Sort and keep best unseen candidates
    best = candidates.sort_values("predicted_Jfinal", ascending=False).head(top_k).copy()

    best = best.drop(columns=["config_key"])
    best.to_csv("offline_suggested_candidates.csv", index=False)

    print("\n=== Top 5 model-suggested NEW candidate configurations ===")
    show_cols = ["predicted_Jfinal"] + FEATURES
    print(best[show_cols].to_string(index=False))

    return best


def print_ns3_commands(best: pd.DataFrame, start_rng_run: int) -> None:
    print("\n=== ns-3 commands for suggested candidates (3D) ===")
    for offset, (_, row) in enumerate(best.iterrows()):
        rng = start_rng_run + offset
        cmd = [
            './ns3 run "scratch/uav-optimisation-3d',  # <-- hyphen matches your .cc filename
            f'--RngRun={rng}',
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

    best = find_best_candidates(model, df_existing=df, top_k=5)

    # Set the first new RNG run as the next unused number
    start_rng_run = int(df["rngRun"].max()) + 1

    # Print one command per suggested candidate, incrementing RNG progressively
    print_ns3_commands(best, start_rng_run=start_rng_run)

    print("\nSaved:")
    print(" - offline_ranked_runs.csv")
    print(" - offline_feature_importance.csv")
    print(" - offline_suggested_candidates.csv")


if __name__ == "__main__":
    main()

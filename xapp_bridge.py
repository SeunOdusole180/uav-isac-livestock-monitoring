# ============================================================
# xapp_bridge.py
# Near-RT RIC xApp bridge — connects to ns-3 via UDP sockets
# 
# Workflow:
#   1. Loads pre-trained RF model (offline_rf_model.pkl)
#   2. Listens on UDP port 5555 for KPI reports from ns-3
#   3. Runs a closed-loop control cycle every 1 second
#   4. Sends parameter updates back to ns-3 on UDP port 5556
#
# Run:  python xapp_bridge.py
# ============================================================

import socket
import json
import time
import logging
import threading
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("UAV-xApp-Bridge")

# ============================================================
# CONFIGURATION
# ============================================================

NS3_KPI_LISTEN_PORT  = 5555    # xApp listens for KPIs from ns-3
NS3_CTRL_SEND_PORT   = 5556    # xApp sends param updates to ns-3
NS3_HOST             = "127.0.0.1"

CONTROL_INTERVAL_S   = 1.0
MODEL_PATH           = "offline_rf_model.pkl"

POLICY = {
    "target_pdet":      0.97,
    "target_rmse_m":    25.0,
    "max_energy_j":     350000.0,
    "target_delay_ms":  20.0,
    "target_loss_pct":   5.0,
}

BOUNDS = {
    "survSenseDt":  (0.3,  3.0),
    "patSenseDt":   (0.3,  3.0),
    "rapidSenseDt": (0.3,  3.0),
    "stratSenseDt": (0.3,  3.0),
    "survVmax":     (5.0,  25.0),
    "patVmax":      (5.0,  35.0),
    "rapidVmax":    (10.0, 45.0),
    "stratVmax":    (8.0,  30.0),
    "survZmin":     (50.0, 70.0),
    "survZmax":     (70.0, 90.0),
    "patZmin":      (70.0, 90.0),
    "patZmax":      (90.0, 110.0),
    "rapidZmin":    (90.0, 110.0),
    "rapidZmax":    (110.0, 130.0),
    "stratZmin":    (110.0, 130.0),
    "stratZmax":    (130.0, 150.0),
    "survDt":       (0.5,  3.0),
    "patDt":        (0.5,  3.0),
    "rapidDt":      (0.5,  2.5),
    "stratDt":      (0.5,  3.0),
}

# These must match the FEATURES list in offline_analysis.py
FEATURES = [
    "survSenseDt", "patSenseDt", "rapidSenseDt", "stratSenseDt",
    "alphaSurv", "alphaPat", "alphaRapid", "alphaStrat",
    "survVmin", "survVmax", "patVmin", "patVmax",
    "rapidVmin", "rapidVmax", "stratVmin", "stratVmax",
    "survZmin", "survZmax", "patZmin", "patZmax",
    "rapidZmin", "rapidZmax", "stratZmin", "stratZmax",
    "survDt", "patDt", "rapidDt", "stratDt"
]

# ============================================================
# SHARED STATE
# ============================================================

_lock      = threading.Lock()
_kpis      = {}
_params    = {
    "survSenseDt":  1.0,
    "patSenseDt":   1.0,
    "rapidSenseDt": 1.0,
    "stratSenseDt": 1.0,
    "alphaSurv":    0.80,
    "alphaPat":     0.60,
    "alphaRapid":   0.35,
    "alphaStrat":   0.90,
    "survVmin":     8.0,
    "survVmax":     15.0,
    "patVmin":      15.0,
    "patVmax":      25.0,
    "rapidVmin":    20.0,
    "rapidVmax":    35.0,
    "stratVmin":    12.0,
    "stratVmax":    22.0,
    "survZmin":     60.0,
    "survZmax":     80.0,
    "patZmin":      80.0,
    "patZmax":      95.0,
    "rapidZmin":    95.0,
    "rapidZmax":    110.0,
    "stratZmin":    120.0,
    "stratZmax":    130.0,
    "survDt":       2.0,
    "patDt":        1.5,
    "rapidDt":      1.0,
    "stratDt":      1.2,
}
_kpi_event = threading.Event()
_cycle     = 0

# ============================================================
# KPI RECEIVER THREAD
# Listens on UDP port 5555 for KPI JSON from ns-3
# ============================================================

def kpi_receiver_thread():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", NS3_KPI_LISTEN_PORT))
    sock.settimeout(5.0)
    logger.info("KPI receiver listening on UDP port %d",
                NS3_KPI_LISTEN_PORT)

    while True:
        try:
            data, addr = sock.recvfrom(4096)
            msg = json.loads(data.decode())
            with _lock:
                _kpis.update(msg)
                _kpis["_received_at"] = time.time()
            _kpi_event.set()
            logger.info(
                "KPI received: Pdet=%.3f  RMSE=%.1fm  "
                "Energy=%.0fJ  sim_t=%.1fs",
                msg.get("pdet",     0.0),
                msg.get("rmse_m",   0.0),
                msg.get("energy_j", 0.0),
                msg.get("sim_time", 0.0),
            )
        except socket.timeout:
            continue
        except Exception as e:
            logger.error("KPI receive error: %s", e)

# ============================================================
# MODEL LOADER
# ============================================================

def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("RF model loaded from %s", MODEL_PATH)
        return model
    except FileNotFoundError:
        logger.warning(
            "RF model not found at %s — "
            "using rule-based control only", MODEL_PATH
        )
        return None

# ============================================================
# PARAMETER CLIPPING
# ============================================================

def clip_params(params):
    clipped = {}
    for k, v in params.items():
        if k in BOUNDS:
            lo, hi = BOUNDS[k]
            clipped[k] = float(np.clip(v, lo, hi))
        else:
            clipped[k] = v
    return clipped

# ============================================================
# RULE-BASED CONTROL DECISIONS
# ============================================================

def rule_based_update(kpis, params):
    updates = {}
    reasons = []

    pdet      = kpis.get("pdet",       0.0)
    rmse      = kpis.get("rmse_m",     999.0)
    energy    = kpis.get("energy_j",   0.0)
    thr       = kpis.get("throughput", 0.0)
    delay     = kpis.get("delay_ms",   0.0)
    loss      = kpis.get("loss_pct",   0.0)

    # Compute J-score from live KPIs so xApp can log it
    THR_REF   = 0.0020
    RMSE_REF  = 30.0
    DELAY_REF = 20.0

    import numpy as np
    Pdet_n  = float(np.clip(pdet,           0.0, 1.0))
    Thr_n   = float(np.clip(thr/THR_REF,   0.0, 1.0))
    RMSE_n  = float(np.clip(rmse/RMSE_REF, 0.0, 1.0))
    Delay_n = float(np.clip(delay/DELAY_REF,0.0, 1.0))
    Loss_n  = float(np.clip(loss/100.0,    0.0, 1.0))

    jscore = 3.0*Pdet_n + 3.0*Thr_n - 2.0*RMSE_n - 1.0*Delay_n - 1.0*Loss_n
    logger.info(
        "Cycle KPIs → Pdet=%.3f RMSE=%.1fm Thr=%.5fMbps "
        "Delay=%.1fms Loss=%.1f%% J=%.3f",
        pdet, rmse, thr, delay, loss, jscore
    )

    # Rule 1: Detection too low → increase sensing frequency
    if pdet < POLICY["target_pdet"]:
        for k in ["survSenseDt", "patSenseDt",
                  "rapidSenseDt", "stratSenseDt"]:
            updates[k] = params.get(k, 1.0) * 0.85
        reasons.append(
            f"Pdet={pdet:.3f} < {POLICY['target_pdet']:.3f} "
            f"→ sensing intervals ×0.85"
        )

    # Rule 2: RMSE too high → reduce speed and lower altitude
    if rmse > POLICY["target_rmse_m"]:
        for k in ["survVmax", "patVmax", "rapidVmax", "stratVmax"]:
            updates[k] = params.get(k, 20.0) * 0.90
        for k in ["survZmax", "patZmax", "rapidZmax", "stratZmax"]:
            updates[k] = params.get(k, 100.0) * 0.95
        reasons.append(
            f"RMSE={rmse:.1f}m > {POLICY['target_rmse_m']:.1f}m "
            f"→ speed ×0.90, altitude ×0.95"
        )

    # Rule 3: Energy too high → relax sensing and speed
    if energy > POLICY["max_energy_j"]:
        for k in ["survSenseDt", "patSenseDt"]:
            updates[k] = params.get(k, 1.0) * 1.20
        for k in ["survVmax", "patVmax"]:
            updates[k] = params.get(k, 20.0) * 0.90
        reasons.append(
            f"Energy={energy:.0f}J > {POLICY['max_energy_j']:.0f}J "
            f"→ sensing ×1.20, speed ×0.90"
        )


# Rule 4: Delay too high → reduce sensing load
    if delay > POLICY.get("target_delay_ms", 20.0):
        for k in ["survSenseDt", "patSenseDt",
                  "rapidSenseDt", "stratSenseDt"]:
            updates[k] = params.get(k, 1.0) * 1.10
        reasons.append(
            f"Delay={delay:.1f}ms > target "
            f"→ sensing intervals ×1.10 to reduce load"
        )

    # Rule 5: Packet loss too high → reduce sensing packet rate
    if loss > POLICY.get("target_loss_pct", 5.0):
        for k in ["survSenseDt", "patSenseDt"]:
            updates[k] = params.get(k, 1.0) * 1.15
        reasons.append(
            f"Loss={loss:.1f}% > target "
            f"→ sensing intervals ×1.15"
        )


    for r in reasons:
        logger.info("Control decision: %s", r)

    return updates

# ============================================================
# MODEL-GUIDED REFINEMENT
# Uses RF model to search neighbourhood of current params
# ============================================================

def model_guided_update(model, params, cycle):
    if model is None:
        return {}

    rng = np.random.default_rng(cycle)
    n   = 500

    candidates = {}
    for feat in FEATURES:
        if feat in BOUNDS:
            lo, hi = BOUNDS[feat]
        else:
            # For features not in BOUNDS (e.g. alpha, Vmin)
            # use a small perturbation around current value
            centre = params.get(feat, 0.5)
            lo, hi = centre * 0.8, centre * 1.2

        centre = params.get(feat, (lo + hi) / 2)
        spread = (hi - lo) * 0.10
        candidates[feat] = np.clip(
            rng.normal(centre, spread, size=n), lo, hi
        )

    cand_df = pd.DataFrame(candidates)

    # Enforce Zmax > Zmin
    for role in ["surv", "pat", "rapid", "strat"]:
        cand_df[f"{role}Zmax"] = np.maximum(
            cand_df[f"{role}Zmax"],
            cand_df[f"{role}Zmin"] + 5.0
        )

    scores = model.predict(cand_df[FEATURES])
    best_idx = int(np.argmax(scores))

    # Score of current params
    current_row = pd.DataFrame(
        [{f: params.get(f, 0) for f in FEATURES}]
    )
    base_score = model.predict(current_row)[0]

    improvement = scores[best_idx] - base_score
    if improvement < 0.01:
        logger.debug(
            "Cycle %d: model improvement %.4f < 0.01 — skipping",
            cycle, improvement
        )
        return {}

    logger.info(
        "Cycle %d: model-guided update — "
        "predicted J improvement +%.4f",
        cycle, improvement
    )
    return {f: float(cand_df.iloc[best_idx][f]) for f in FEATURES}

# ============================================================
# SEND PARAM UPDATE TO ns-3
# ============================================================

def send_params(params):
    # Only send the parameters that ns-3 actually uses
    # (excludes alpha, Vmin which are fixed at startup)
    sendable_keys = [
        "survSenseDt", "patSenseDt", "rapidSenseDt", "stratSenseDt",
        "survVmax",    "patVmax",    "rapidVmax",    "stratVmax",
        "survZmin",    "survZmax",   "patZmin",      "patZmax",
        "rapidZmin",   "rapidZmax",  "stratZmin",    "stratZmax",
        "survDt",      "patDt",      "rapidDt",      "stratDt",
    ]
    payload_dict = {k: params[k] for k in sendable_keys
                    if k in params}
    payload = json.dumps(payload_dict).encode()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(payload, (NS3_HOST, NS3_CTRL_SEND_PORT))
    sock.close()
    logger.info(
        "Param update sent to ns-3:%d — %d params",
        NS3_CTRL_SEND_PORT, len(payload_dict)
    )

# ============================================================
# MAIN CONTROL LOOP
# ============================================================

def main():
    global _cycle

    model = load_model()

    # Start KPI receiver in background thread
    t = threading.Thread(
        target=kpi_receiver_thread,
        daemon=True,
        name="KpiReceiver"
    )
    t.start()

    logger.info(
        "xApp bridge running — "
        "waiting for KPI reports from ns-3 on port %d",
        NS3_KPI_LISTEN_PORT
    )

    while True:
        _cycle += 1
        start = time.monotonic()

        # Wait up to CONTROL_INTERVAL_S for a new KPI
        got_kpi = _kpi_event.wait(timeout=CONTROL_INTERVAL_S)
        _kpi_event.clear()

        if not got_kpi:
            logger.debug(
                "Cycle %d: no KPI received in %.1fs — waiting",
                _cycle, CONTROL_INTERVAL_S
            )
            continue

        with _lock:
            kpis   = dict(_kpis)
            params = dict(_params)

        # Rule-based decisions
        updates = rule_based_update(kpis, params)

        # Model-guided refinement every 5 cycles
        if _cycle % 5 == 0:
            model_updates = model_guided_update(
                model, params, _cycle
            )
            updates.update(model_updates)

        if updates:
            with _lock:
                _params.update(updates)
                clipped = clip_params(dict(_params))
                _params.update(clipped)
                new_params = dict(_params)

            send_params(new_params)
        else:
            logger.info(
                "Cycle %d: all KPIs within targets — "
                "no update sent", _cycle
            )

        # Pace the loop
        elapsed = time.monotonic() - start
        time.sleep(max(0.0, CONTROL_INTERVAL_S - elapsed))


if __name__ == "__main__":
    main()
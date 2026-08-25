from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import os
import glob
import json

# Adjust imports to work when run from backend module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.generator import SimulatorConfig, generate_events
from simulator.bias import assign_historical_treatments
from causal_engine.aipw_estimator import AIPWEstimator
from causal_engine.estimator import TLearner  # kept for benchmark comparison only
from policies.baseline import GrossRecoveryBaseline
from policies.incrementality import IncrementalityAwarePolicy
from policies.oracle import OraclePolicy
from evaluation.evaluator import evaluate_policy

app = FastAPI(title="Razorpay CausaPay Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global objects – initialized once on server start
# ---------------------------------------------------------------------------

aipw_estimator: AIPWEstimator = None
baseline_policy: GrossRecoveryBaseline = None
inc_policy: IncrementalityAwarePolicy = None
oracle_policy: OraclePolicy = None
train_obs: pd.DataFrame = None
test_obs: pd.DataFrame = None
test_hid: pd.DataFrame = None
events_df: pd.DataFrame = None  # stripped version for API exposure
benchmark_results: list = []

costs = {"none": 0.0, "retry": 2.0, "whatsapp": 15.0}


def _strip_hidden(df: pd.DataFrame) -> pd.DataFrame:
    """Remove any hidden ground‑truth columns before JSON serialisation.
    Columns that start with 'true_' or 'Y_' are considered hidden.
    """
    hidden = [c for c in df.columns if c.startswith("true_") or c.startswith("Y_")]
    return df.drop(columns=hidden, errors="ignore")


def _build_decision_response(row: pd.Series) -> dict:
    """Given a row (already containing probabilities and stds),
    compute incremental probabilities, ENIV, and recommendation.
    This mirrors IncrementalityAwarePolicy logic for a single observation.
    """
    p_none = row["prob_none"]
    p_retry = row["prob_retry"]
    p_wa = row["prob_whatsapp"]
    std_retry = row.get("std_retry", np.nan)
    std_wa = row.get("std_whatsapp", np.nan)
    amount = row["amount"]

    inc_prob_retry = p_retry - p_none
    inc_prob_wa = p_wa - p_none
    eniv_none = 0.0
    eniv_retry = inc_prob_retry * amount - costs["retry"]
    eniv_wa = inc_prob_wa * amount - costs["whatsapp"]

    # uncertainty abstention
    if std_retry > inc_policy.uncertainty_threshold or std_wa > inc_policy.uncertainty_threshold:
        recommended = baseline_policy.predict(pd.DataFrame([row]))[0]
        is_abstain = True
    else:
        is_abstain = False
        best_eniv = max(eniv_none, eniv_retry, eniv_wa)
        if best_eniv <= 0:
            recommended = "none"
        elif best_eniv == eniv_retry:
            recommended = "retry"
        else:
            recommended = "whatsapp"

    explanation = (
        f"ENIV (none)={eniv_none:.2f}, ENIV (retry)={eniv_retry:.2f}, ENIV (whatsapp)={eniv_wa:.2f}. "
        f"Recommendation: {recommended}."
    )

    return {
        "event_id": str(row["event_id"]),
        "prob_none": float(p_none),
        "prob_retry": float(p_retry),
        "prob_whatsapp": float(p_wa),
        "inc_prob_retry": float(inc_prob_retry),
        "inc_prob_whatsapp": float(inc_prob_wa),
        "eniv_none": float(eniv_none),
        "eniv_retry": float(eniv_retry),
        "eniv_whatsapp": float(eniv_wa),
        "recommended_action": recommended,
        "is_abstain": is_abstain,
        "explanation": explanation,
    }


def _event_payload(row: pd.Series) -> dict:
    """Return the observable fields needed to browse a real simulated event."""
    fields = [
        "event_id", "customer_id", "amount", "plan_tier", "payment_method",
        "failure_context", "decline_signal_bucket", "engagement_score",
        "historical_failure_count", "whatsapp_opted_in", "recommended_action",
        "baseline_action", "is_abstain",
    ]
    payload = {field: row[field] for field in fields if field in row.index}
    if "event_id" in payload:
        payload["event_id"] = str(payload["event_id"])
    return {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in payload.items()
    }


@app.on_event("startup")
async def load_models_and_data():
    global aipw_estimator, baseline_policy, inc_policy, oracle_policy
    global train_obs, test_obs, test_hid, events_df, benchmark_results

    # 1. generate ground truth
    config = SimulatorConfig(seed=42, num_events=15000)
    obs, hid = generate_events(config)

    # 2. assign historical treatments (selection bias)
    obs_biased = assign_historical_treatments(obs, hid, seed=42)

    # 3. train / test split (70/30)
    train_size = int(0.7 * len(obs_biased))
    train_obs = obs_biased.iloc[:train_size].copy()
    test_obs = obs_biased.iloc[train_size:].copy()
    test_hid = hid.iloc[train_size:].copy()

    # 4. train models
    aipw_estimator = AIPWEstimator()
    aipw_estimator.fit(train_obs)
    baseline_policy = GrossRecoveryBaseline()
    baseline_policy.fit(train_obs)
    inc_policy = IncrementalityAwarePolicy(aipw_estimator, baseline_policy)
    oracle_policy = OraclePolicy()

    # 5. evaluate policies (store for comparison)
    baseline_actions = baseline_policy.predict(test_obs)
    inc_results = inc_policy.predict(test_obs)
    inc_actions = inc_results["recommended_action"].values
    oracle_results = oracle_policy.predict(test_obs, test_hid)
    oracle_actions = oracle_results["recommended_action"].values

    res_baseline = evaluate_policy("Baseline (Gross Recovery)", baseline_actions, test_obs, test_hid)
    res_inc = evaluate_policy("Incrementality-Aware", inc_actions, test_obs, test_hid)
    res_oracle = evaluate_policy("Oracle", oracle_actions, test_obs, test_hid)
    benchmark_results = [res_baseline, res_inc, res_oracle]

    # 6. prepare events dataframe for API exposure
    cf = aipw_estimator.predict_counterfactuals(test_obs)
    events = test_obs.reset_index(drop=True)
    cf = cf.reset_index(drop=True)
    events = pd.concat([events, cf.drop(columns=["event_id"])], axis=1)
    events["baseline_action"] = baseline_actions
    # `events` has a reset index whereas inc_results keeps the test-set index.
    # Assign raw values to avoid pandas label alignment producing NaN actions.
    events["recommended_action"] = inc_results["recommended_action"].values
    events["is_abstain"] = inc_results["is_abstain"].values
    amount = events["amount"].values
    inc_prob_retry = events["prob_retry"] - events["prob_none"]
    inc_prob_wa = events["prob_whatsapp"] - events["prob_none"]
    events["inc_prob_retry"] = inc_prob_retry
    events["inc_prob_whatsapp"] = inc_prob_wa
    events["eniv_retry"] = inc_prob_retry * amount - costs["retry"]
    events["eniv_wa"] = inc_prob_wa * amount - costs["whatsapp"]

    # Store stripped version for API (remove hidden ground‑truth columns)
    events_df = _strip_hidden(events)
    app.state.full_events = events  # keep full version internally
    app.state.events_lookup = {str(row["event_id"]): row for _, row in events_df.iterrows()}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class DecisionRequest(BaseModel):
    event_id: str = Field(..., description="Unique identifier for the failed payment")
    amount: float
    class Config:
        extra = "allow"

class DecisionResponse(BaseModel):
    event_id: str
    prob_none: float
    prob_retry: float
    prob_whatsapp: float
    inc_prob_retry: float
    inc_prob_whatsapp: float
    eniv_none: float
    eniv_retry: float
    eniv_whatsapp: float
    recommended_action: str
    is_abstain: bool
    explanation: str

class SummaryResponse(BaseModel):
    total_failed_payments: int
    gross_recovered: float
    incremental_recovered: float
    intervention_cost: float
    policy_value: float
    action_distribution: dict

class PolicyResult(BaseModel):
    policy_name: str
    gross_recovered: float
    true_incremental_recovered: float
    intervention_cost: float
    policy_value: float
    recovery_rate: float
    action_distribution: dict


class EventListResponse(BaseModel):
    events: list[dict]
    total: int

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    return {"status": "ok", "model_loaded": aipw_estimator is not None}

@app.get("/api/summary", response_model=SummaryResponse)
def get_summary():
    if events_df is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    full = app.state.full_events
    actions = events_df["recommended_action"].values
    true_y = np.zeros(len(actions))
    for i, a in enumerate(actions):
        true_y[i] = test_hid[f"Y_{a}"].iloc[i]
    amounts = full["amount"].values
    gross = (true_y * amounts).sum()
    true_none = test_hid["Y_none"].values
    incremental = ((true_y - true_none) * amounts).sum()
    total_cost = sum(costs.get(a, 0.0) for a in actions)
    policy_val = incremental - total_cost
    distribution = pd.Series(actions).value_counts().to_dict()
    return SummaryResponse(
        total_failed_payments=len(events_df),
        gross_recovered=round(gross, 2),
        incremental_recovered=round(incremental, 2),
        intervention_cost=round(total_cost, 2),
        policy_value=round(policy_val, 2),
        action_distribution=distribution,
    )

@app.post("/api/decision", response_model=DecisionResponse)
def make_decision(req: DecisionRequest):
    if aipw_estimator is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    # Existing event IDs are resolved against the real observable event data.
    # For a new event, callers must supply the estimator feature fields in addition
    # to event_id and amount; pandas/estimator validation will return a clear 422.
    data = req.dict()
    existing = getattr(app.state, "events_lookup", {}).get(req.event_id)
    if existing is not None:
        merged = existing.to_dict()
        merged.update({key: value for key, value in data.items() if value is not None})
        data = merged
    df = pd.DataFrame([data])
    try:
        cf = aipw_estimator.predict_counterfactuals(df)
    except KeyError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required observable feature: {exc.args[0]}",
        ) from exc
    row = df.iloc[0].copy()
    # Both frames contain event_id; assigning prediction columns avoids duplicate
    # labels, which would make scalar decision calculations ambiguous.
    for column, value in cf.iloc[0].items():
        if column != "event_id":
            row[column] = value
    response = _build_decision_response(row)
    return DecisionResponse(**response)

@app.get("/api/policy-comparison", response_model=list[PolicyResult])
def get_policy_comparison():
    if not benchmark_results:
        raise HTTPException(status_code=503, detail="Benchmark not ready")
    return [PolicyResult(**r) for r in benchmark_results]

@app.get("/api/events/{event_id}")
def get_event(event_id: str):
    if events_df is None:
        raise HTTPException(status_code=503, detail="Data not ready")
    row = app.state.events_lookup.get(event_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Event not found")
    response = _build_decision_response(row)
    response["baseline_action"] = row["baseline_action"]
    response["is_abstain"] = row["is_abstain"]
    return response


@app.get("/api/events", response_model=EventListResponse)
def list_events(limit: int = 30):
    """Browse real, observable held-out events without exposing hidden outcomes."""
    if events_df is None:
        raise HTTPException(status_code=503, detail="Data not ready")
    limit = max(1, min(limit, 100))
    rows = [_event_payload(row) for _, row in events_df.head(limit).iterrows()]
    return EventListResponse(events=rows, total=len(events_df))

@app.get("/api/diagnostics")
def get_diagnostics():
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation", "results")
    pattern = os.path.join(results_dir, "diagnostic_summary_*.json")
    files = sorted(glob.glob(pattern), reverse=True)
    if not files:
        raise HTTPException(status_code=404, detail="No diagnostic file found")
    latest = files[0]
    with open(latest, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

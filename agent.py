import json
import os
import random
import re
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from hindsight_client import Hindsight
from openai import OpenAI

from runner import ALLOWED, load, run

load_dotenv()

SESSIONS = 3
RUNS_PER_SESSION = 10
MODEL = "openai/gpt-oss-120b"
LIMITS = {
    "n_estimators": (10, 300),
    "max_depth": (1, 40),
    "C": (0.0001, 100.0),
    "learning_rate": (0.01, 1.0),
}

llm = OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")
columns = list(load()[0].columns)

SYSTEM = f"""You tune a scikit-learn classifier that predicts customer churn. The metric is test F1 on the churn class; higher is better.
Reply with one JSON object and nothing else, in this shape:
{{"model": "logreg" | "random_forest" | "gradient_boosting", "params": {{...}}, "drop_columns": [...], "scale": true | false}}
Allowed params per model: logreg: C. random_forest: n_estimators, max_depth. gradient_boosting: n_estimators, learning_rate, max_depth.
Available columns you may drop: {columns}
A run is marked overfit when train F1 exceeds test F1 by more than 0.15."""


def clamp(config):
    model = config.get("model")
    if model not in ALLOWED:
        raise ValueError("unknown model")
    params = {}
    for key, value in (config.get("params") or {}).items():
        if key in ALLOWED[model]:
            low, high = LIMITS[key]
            params[key] = min(max(float(value), low), high)
    drops = [c for c in (config.get("drop_columns") or []) if c in columns]
    return {"model": model, "params": params, "drop_columns": drops, "scale": bool(config.get("scale", True))}


def random_config():
    model = random.choice(list(ALLOWED))
    params = {k: random.uniform(*LIMITS[k]) for k in ALLOWED[model]}
    return clamp({"model": model, "params": params, "drop_columns": [], "scale": True})


def propose(session_log, recalled):
    parts = []
    if recalled:
        parts.append("Lessons from earlier sessions:\n" + "\n".join(f"- {t}" for t in recalled))
    if session_log:
        parts.append(
            "Runs so far in this session:\n"
            + "\n".join(f"- {json.dumps(c)} -> {json.dumps(r)}" for c, r in session_log)
        )
    parts.append("Propose the next configuration to try. Do not repeat a configuration that already failed or overfit.")
    for _ in range(2):
        try:
            reply = llm.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": "\n\n".join(parts)},
                ],
                temperature=0.6,
            )
            match = re.search(r"\{.*\}", reply.choices[0].message.content, re.S)
            return clamp(json.loads(match.group(0))), False
        except Exception:
            time.sleep(2)
    return random_config(), True


def main(mode, trial):
    memory = mode == "memory"
    client = None
    if memory:
        client = Hindsight(
            base_url=os.environ["HINDSIGHT_BASE_URL"],
            api_key=os.environ["HINDSIGHT_API_KEY"],
        )
    bank = f"ninaivu-{trial}-{int(time.time())}"
    seen_failures = set()
    best = 0.0
    for session in range(1, SESSIONS + 1):
        session_log = []
        rows = []
        for index in range(1, RUNS_PER_SESSION + 1):
            recalled = []
            if memory and session > 1:
                found = client.recall(
                    bank_id=bank,
                    query="Which configurations overfit, errored, or scored best, and what should be tried next?",
                )
                recalled = [r.text for r in found.results[:12]]
            config, fallback = propose(session_log, recalled)
            result = run(config)
            key = json.dumps(config, sort_keys=True)
            repeated = key in seen_failures
            if result["status"] != "ok":
                seen_failures.add(key)
            score = result.get("test_f1", 0.0)
            best = max(best, score)
            session_log.append((config, result))
            if memory:
                client.retain(
                    bank_id=bank,
                    content=f"Experiment: {json.dumps(config)} -> {json.dumps(result)}",
                    context="ML experiment log on churn classifier",
                )
            rows.append({
                "mode": mode,
                "trial": trial,
                "session": session,
                "run": index,
                "test_f1": score,
                "best_so_far": best,
                "status": result["status"],
                "repeated_failure": repeated,
                "fallback": fallback,
                "config": key,
            })
            print(f"{mode} s{session} r{index}: {result['status']} f1={score} best={best}")
        pd.DataFrame(rows).to_csv("results.csv", mode="a", header=not os.path.exists("results.csv"), index=False)
        if memory:
            time.sleep(30)
    if client:
        client.close()


if __name__ == "__main__":
    if sys.argv[1] not in ("memory", "baseline"):
        raise SystemExit("usage: python agent.py memory|baseline TRIAL")
    main(sys.argv[1], sys.argv[2])
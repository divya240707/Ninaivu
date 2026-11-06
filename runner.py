import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/dataset.csv"
TARGET = "Churn"
POSITIVE = "Yes"
SEED = 42

ALLOWED = {
    "logreg": {"C": float},
    "random_forest": {"n_estimators": int, "max_depth": int},
    "gradient_boosting": {"n_estimators": int, "learning_rate": float, "max_depth": int},
}


def load():
    df = pd.read_csv(DATA_PATH)
    for col in df.select_dtypes(["object", "str"]):
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() > 0.9:
            df[col] = converted
    y = (df[TARGET] == POSITIVE).astype(int)
    return df.drop(columns=[TARGET]), y


def build_model(name, params):
    if name == "logreg":
        return LogisticRegression(max_iter=1000, **params)
    if name == "random_forest":
        return RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)
    return GradientBoostingClassifier(random_state=SEED, **params)


def run(config):
    try:
        X, y = load()
        X = X.drop(columns=[c for c in config.get("drop_columns", []) if c in X.columns])
        numeric = X.select_dtypes("number").columns.tolist()
        categorical = [c for c in X.columns if c not in numeric]
        allowed = ALLOWED[config["model"]]
        params = {k: allowed[k](v) for k, v in config.get("params", {}).items() if k in allowed}
        scaler = StandardScaler() if config.get("scale", True) else "passthrough"
        pre = ColumnTransformer([
            ("num", Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", scaler)]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ])
        pipe = Pipeline([("pre", pre), ("model", build_model(config["model"], params))])
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
        pipe.fit(X_tr, y_tr)
        train_f1 = f1_score(y_tr, pipe.predict(X_tr))
        test_f1 = f1_score(y_te, pipe.predict(X_te))
        gap = train_f1 - test_f1
        return {
            "status": "overfit" if gap > 0.15 else "ok",
            "test_f1": round(test_f1, 4),
            "train_f1": round(train_f1, 4),
            "gap": round(gap, 4),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)[:200]}


if __name__ == "__main__":
    print(run({"model": "logreg", "params": {"C": 1.0}, "drop_columns": ["customerID"]}))
    print(run({"model": "random_forest", "params": {"n_estimators": 200, "max_depth": 30}}))
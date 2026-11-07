import pandas as pd

d = pd.read_csv("results.csv")
d["dropped_id"] = d["config"].str.contains("customerID")
print(d.groupby(["mode", "session"])["dropped_id"].sum())
print(d.groupby("dropped_id")["test_f1"].agg(["mean", "count"]))
print(d.groupby("dropped_id")["status"].apply(lambda s: (s != "ok").mean()))
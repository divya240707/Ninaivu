import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")
df["bad_run"] = df["status"] != "ok"

summary = (
    df.groupby(["mode", "session"])
    .agg(
        mean_f1=("test_f1", "mean"),
        bad_runs=("bad_run", "sum"),
        repeated_failures=("repeated_failure", "sum"),
        fallbacks=("fallback", "sum"),
        trials=("trial", "nunique"),
    )
    .round(4)
)
print(summary)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for mode, group in df.groupby("mode"):
    f1 = group.groupby(["trial", "session"])["test_f1"].mean().groupby("session").agg(["mean", "std"]).fillna(0)
    axes[0].errorbar(f1.index, f1["mean"], yerr=f1["std"], marker="o", capsize=3, label=mode)
    bad = group.groupby(["trial", "session"])["bad_run"].sum().groupby("session").mean()
    axes[1].plot(bad.index, bad.values, marker="o", label=mode)

axes[0].set_title("Mean test F1 per session")
axes[1].set_title("Overfit or error runs per session")
for ax in axes:
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Session")
    ax.legend()
plt.tight_layout()
plt.savefig("comparison.png", dpi=200)
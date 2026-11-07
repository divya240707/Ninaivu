import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from hindsight_client import Hindsight

load_dotenv()

BANK = "ninaivu-demo"
path = sys.argv[1] if len(sys.argv) > 1 else "results_v1.csv"

df = pd.read_csv(path)
memory_rows = df[df["mode"] == "memory"]
first_trial = memory_rows["trial"].iloc[0]
rows = memory_rows[memory_rows["trial"] == first_trial]

manual_notes = [
    "Manual check: logistic regression with C=1.0 and the customerID column kept got train F1 0.7543 and test F1 0.6047, a train-test gap of 0.1496.",
    "Manual check: logistic regression with C=1.0 and the customerID column dropped got train F1 0.6031 and test F1 0.607, a gap of about zero, so dropping customerID removed the overfitting.",
    "Manual check: random forest with 200 trees and max_depth 30 got train F1 0.8857 and test F1 0.5371, a gap of 0.3487, so it was flagged as overfit.",
]

client = Hindsight(
    base_url=os.environ["HINDSIGHT_BASE_URL"],
    api_key=os.environ["HINDSIGHT_API_KEY"],
)

for note in manual_notes:
    client.retain(bank_id=BANK, content=note, context="manual check of the runner on the churn dataset")
    print("retained manual note")

for _, row in rows.iterrows():
    content = (
        f"Session {row['session']}, run {row['run']}: configuration {row['config']} "
        f"finished with status {row['status']} and test F1 {row['test_f1']}."
    )
    client.retain(bank_id=BANK, content=content, context="ML experiment log on churn classifier")
    print(f"retained session {row['session']} run {row['run']}")

time.sleep(60)
client.close()
print("done")
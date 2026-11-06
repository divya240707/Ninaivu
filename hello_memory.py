import os
import time
from dotenv import load_dotenv
from hindsight_client import Hindsight

load_dotenv()
client = Hindsight(
    base_url=os.environ["HINDSIGHT_BASE_URL"],
    api_key=os.environ["HINDSIGHT_API_KEY"],
)

BANK = "ninaivu-smoke-test"

client.retain(
    bank_id=BANK,
    content="Run 1: random_forest with max_depth=30 scored 0.71 test F1 but 1.00 train F1, so it overfit.",
    context="experiment log",
)
time.sleep(8)

answer = client.reflect(bank_id=BANK, query="What should I avoid next time?")
print(answer.text)
client.close()
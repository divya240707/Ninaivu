import os

import pandas as pd
import altair as alt
import streamlit as st
from dotenv import load_dotenv
from hindsight_client import Hindsight

load_dotenv()

BANK = "ninaivu-demo"
RESULTS_FILE = "results.csv"
EXAMPLES = [
    "Why should we drop the customerID column?",
    "Which configurations overfit?",
    "What is the best configuration so far?",
]

st.set_page_config(page_title="Ninaivu", layout="wide")

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-size: 19px; }
    .block-container { padding-top: 2.2rem; max-width: 1500px; }
    h1 { font-size: 3.4rem !important; }
    h2, h3 { font-size: 1.9rem !important; }
    div[data-testid="stMetricValue"] { font-size: 2.8rem; }
    div[data-testid="stMetricLabel"] p { font-size: 1.15rem; }
    button[data-baseweb="tab"] p { font-size: 1.25rem; }
    div[data-testid="stCaptionContainer"] p { font-size: 1.05rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_client():
    return Hindsight(
        base_url=os.environ["HINDSIGHT_BASE_URL"],
        api_key=os.environ["HINDSIGHT_API_KEY"],
    )


@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_FILE)

def line_chart(wide, y_title, y_domain=None):
    long = wide.reset_index().melt("session", var_name="agent", value_name="value")
    scale = alt.Scale(domain=y_domain, zero=False) if y_domain else alt.Scale(zero=True)
    axis = alt.Axis(labelFontSize=16, titleFontSize=16)
    return (
        alt.Chart(long)
        .mark_line(point=alt.OverlayMarkDef(size=160), strokeWidth=4)
        .encode(
            x=alt.X("session:O", title="Session", axis=alt.Axis(labelAngle=0, labelFontSize=16, titleFontSize=16)),
            y=alt.Y("value:Q", title=y_title, scale=scale, axis=axis),
            color=alt.Color(
                "agent:N",
                scale=alt.Scale(domain=["baseline", "memory"], range=["#1f77b4", "#ff7f0e"]),
                legend=alt.Legend(labelFontSize=16, titleFontSize=16, orient="top"),
            ),
        )
        .properties(height=420)
    )

df = load_results()
df["bad_run"] = df["status"] != "ok"
trials = df.groupby("mode")["trial"].nunique()

per_session = (
    df.groupby(["mode", "session"])
    .agg(mean_f1=("test_f1", "mean"), bad=("bad_run", "sum"), fallbacks=("fallback", "sum"))
    .reset_index()
)
per_session["bad_per_trial"] = per_session.apply(lambda r: r["bad"] / trials[r["mode"]], axis=1)
f1_wide = per_session.pivot(index="session", columns="mode", values="mean_f1")
bad_wide = per_session.pivot(index="session", columns="mode", values="bad_per_trial")

later = df[df["session"] > 1]
later_bad = later.groupby("mode")["bad_run"].sum() / later.groupby("mode")["trial"].nunique()

st.title("Ninaivu")
st.markdown("### An ML lab notebook that remembers every experiment, built on Hindsight memory")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Runs analysed", len(df))
c2.metric("Trials (memory / baseline)", f"{trials['memory']} / {trials['baseline']}")
c3.metric("Fallback runs", int(df["fallback"].sum()))
c4.metric(
    "Bad runs per trial, sessions 2 and 3",
    f"{later_bad['memory']:.1f} vs {later_bad['baseline']:.1f}",
    help="Memory agent vs baseline agent. A bad run is an overfit or error run.",
)

results_tab, ask_tab, runs_tab = st.tabs(["Results", "Ask the memory", "Run history"])

with results_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Mean test F1 per session")
        st.altair_chart(
            line_chart(f1_wide, "Mean test F1", [max(0, f1_wide.min().min() - 0.05), f1_wide.max().max() + 0.05]),
            use_container_width=True,
        )
        gap = (f1_wide["memory"] - f1_wide["baseline"]).abs().max()
        st.caption(f"Largest gap between the two agents in any session: {gap:.3f} F1. The y-axis is zoomed in so both lines are visible.")
    with right:
        st.subheader("Overfit or error runs per trial")
        st.altair_chart(line_chart(bad_wide, "Overfit or error runs per trial"), use_container_width=True)
        st.caption("Neither agent has memory in session 1, so any gap there shows the noise level.")
    st.info(
        f"{int(df['fallback'].sum())} of {len(df)} runs used a random fallback configuration "
        "because the language model call failed."
    )
    st.subheader("Summary table")
    st.dataframe(per_session.round(4), hide_index=True)

with ask_tab:
    st.subheader("Ask the experiment memory")
    if "question" not in st.session_state:
        st.session_state.question = EXAMPLES[0]
    columns = st.columns(len(EXAMPLES))
    for column, example in zip(columns, EXAMPLES):
        if column.button(example):
            st.session_state.question = example
    question = st.text_input("Your question", key="question")
    if st.button("Ask", type="primary") and question.strip():
        client = get_client()
        with st.spinner("Searching memory..."):
            try:
                guarded = (
                    f"{question}\n\nAnswer in at most four sentences using only facts stored in memory. "
                    "Quote the numbers from the stored experiments. If memory does not cover something, say so."
                )
                answer = client.reflect(bank_id=BANK, query=guarded)
                evidence = client.recall(bank_id=BANK, query=question)
            except Exception as exc:
                st.error(f"Memory request failed: {exc}")
                st.stop()
        st.markdown("#### Answer (generated from memory, check it against the evidence below)")
        st.write(answer.text)
        st.markdown("#### Stored memories retrieved for this question")
        for item in evidence.results[:8]:
            st.write(f"- {item.text}")

with runs_tab:
    mode = st.selectbox("Mode", ["memory", "baseline"])
    st.dataframe(
        df[df["mode"] == mode][["trial", "session", "run", "test_f1", "status", "fallback", "config"]],
        height=600,
        hide_index=True,
    )
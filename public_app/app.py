# Ask the Fleet: public demo of the fleet reliability semantic layer, served from Databricks.
# Hosted on Streamlit Community Cloud. Visitors ask questions in plain English; a Databricks
# Genie space turns them into SQL against Unity Catalog metric views and runs it as a read-only
# service principal on the workspace's serverless SQL warehouse.
import json
import os
import sys
import time
import uuid
from datetime import timedelta

import altair as alt
import pandas as pd
import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dashboards import MessageStatus
from databricks.sdk.service.sql import StatementParameterListItem, StatementState

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "streamlit_app"))
from report import METRIC_VIEWS, render_report  # noqa: E402
from modeling import render_modeling  # noqa: E402

REPO_URL = "https://github.com/livingdw67/cmapss-predictive-maintenance"
TOUR_URL = "https://claude.ai/artifact/Ai1U6EospZRsTgpSU9TxBL#tour"
QUESTION_LOG = "workspace.cmapss.demo_question_log"
# The Fleet report is served from a snapshot of the metric views (databricks/export_report_snapshot.py),
# so it keeps working when the Free Edition compute quota runs out; only questions use live compute
SNAPSHOT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_snapshot.json")
# Databricks Free Edition has a daily compute quota shared with the pipeline, so questions are capped
DAILY_LIMIT = 50
SESSION_LIMIT = 10
MAX_QUESTION_CHARS = 300
MAX_ROWS = 1000
SUGGESTIONS = [
    "What is the mean time to failure for each fault mode?",
    "Which dataset has the most variable engine life?",
    "How many engines failed before 150 cycles?",
    "Which engine lasted the longest, and for how many cycles?",
    "How much does HPC outlet temperature rise before failure in FD001?",
    "What share of cycles were flown within 15 cycles of failure?",
]
NUMERIC_TYPES = {"BYTE", "SHORT", "INT", "LONG", "FLOAT", "DOUBLE", "DECIMAL"}


class SetupError(Exception):
    """A configuration problem the app owner can fix; the message never contains secret values."""


@st.cache_resource
def get_client():
    # Each step fails with its own message so a misconfigured deploy says what to fix
    try:
        has_section = "databricks" in st.secrets
    except Exception:  # no secrets configured at all
        has_section = False
    if not has_section:
        raise SetupError("The app's secrets have no [databricks] section. Paste the secrets file "
                         "into the app's Settings > Secrets on Streamlit Community Cloud.")
    cfg = st.secrets["databricks"]
    missing = [k for k in ("host", "client_id", "client_secret", "warehouse_id", "genie_space_id") if not cfg.get(k)]
    if missing:
        raise SetupError(f"The [databricks] secrets are missing: {', '.join(missing)}.")
    try:
        client = WorkspaceClient(host=cfg["host"], client_id=cfg["client_id"], client_secret=cfg["client_secret"])
        client.current_user.me()
        return client
    except Exception as e:
        raise SetupError(f"Databricks rejected the service principal login: {str(e)[:200]}")


def settings():
    return st.secrets["databricks"]


def to_df(response):
    """Statement Execution API response -> DataFrame, with numeric columns typed."""
    if response.status and response.status.state != StatementState.SUCCEEDED:
        error = response.status.error.message if response.status.error else response.status.state.value
        raise RuntimeError(error[:300])
    columns = response.manifest.schema.columns
    rows = response.result.data_array if response.result and response.result.data_array else []
    df = pd.DataFrame(rows, columns=[c.name for c in columns])
    for c in columns:
        if c.type_name and c.type_name.value in NUMERIC_TYPES:
            df[c.name] = pd.to_numeric(df[c.name])
    return df


def execute(sql, parameters=None, max_rows=None):
    client = get_client()
    response = client.statement_execution.execute_statement(
        statement=sql, warehouse_id=settings()["warehouse_id"], parameters=parameters,
        row_limit=max_rows, wait_timeout="50s",
    )
    # A cold warehouse can take longer than the 50-second synchronous wait
    while response.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(1)
        response = client.statement_execution.get_statement(response.statement_id)
    return response


def fetch(sql, max_rows=None):
    return to_df(execute(sql, max_rows=max_rows))


@st.cache_resource
def load_snapshot():
    with open(SNAPSHOT_FILE, encoding="utf-8") as f:
        return json.load(f)


def fetch_snapshot(sql):
    """The report's queries, answered from the metric-view results saved by export_report_snapshot.py."""
    result = load_snapshot()["queries"][sql]
    return pd.DataFrame(result["rows"], columns=result["columns"])


def questions_used_today():
    return int(fetch(f"SELECT COUNT(*) FROM {QUESTION_LOG} WHERE asked_at >= current_date()").iloc[0, 0])


def log_question(question, status, sql, row_count, latency_ms):
    params = [
        StatementParameterListItem(name="session_id", value=st.session_state.session_id),
        StatementParameterListItem(name="question", value=question),
        StatementParameterListItem(name="status", value=status),
        StatementParameterListItem(name="generated_sql", value=sql),
        StatementParameterListItem(name="row_count", value=None if row_count is None else str(row_count), type="INT"),
        StatementParameterListItem(name="latency_ms", value=str(latency_ms), type="INT"),
    ]
    execute(f"INSERT INTO {QUESTION_LOG} (asked_at, session_id, question, status, generated_sql, row_count, latency_ms) "
            "VALUES (current_timestamp(), :session_id, :question, :status, :generated_sql, :row_count, :latency_ms)",
            parameters=params)


def ask_genie(question):
    """Send one question to the Genie space and wait for its answer."""
    return get_client().genie.start_conversation_and_wait(
        settings()["genie_space_id"], question, timeout=timedelta(seconds=120))


def auto_chart(df):
    # Bar chart when the answer is one label column and one number column
    if len(df) < 2 or len(df) > 40 or df.shape[1] != 2:
        return None
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric) != 1:
        return None
    value = numeric[0]
    label = next(c for c in df.columns if c != value)
    return alt.Chart(df).mark_bar(color="#5fc4c0", cornerRadiusEnd=3).encode(
        x=alt.X(f"{value}:Q", title=value.replace("_", " ").lower()),
        y=alt.Y(f"{label}:N", sort="-x", title=None),
        tooltip=[label, alt.Tooltip(f"{value}:Q", format=",.2f")],
    )


def answer(question):
    """Run one question end to end and return a record for the chat history."""
    record = {"question": question, "text": "", "sql": None, "df": None, "suggestions": [], "error": None}
    start = time.time()
    status, sql, row_count = "error", None, None
    try:
        message = ask_genie(question)
        if message.status != MessageStatus.COMPLETED:
            raise RuntimeError(message.error.error if message.error else message.status.value)
        texts = []
        for a in message.attachments or []:
            if a.text and a.text.content:
                texts.append(a.text.content)
            if a.suggested_questions and a.suggested_questions.questions:
                record["suggestions"] = a.suggested_questions.questions
            if a.query and a.query.query and sql is None:
                sql = a.query.query
                if a.query.description:
                    texts.append(a.query.description)
                # Genie already ran the query as the read-only service principal; fetch its result
                result = get_client().genie.get_message_attachment_query_result(
                    settings()["genie_space_id"], message.conversation_id, message.message_id, a.attachment_id)
                df = to_df(result.statement_response).head(MAX_ROWS)
                record["sql"], record["df"] = sql, df
                status, row_count = "ok", len(df)
        record["text"] = "\n\n".join(texts)
        if sql is None:
            status = "no_sql"
    except Exception as e:
        record["error"] = "Something went wrong answering that question. Try rephrasing it."
        record["detail"] = str(e)[:300]
    finally:
        try:
            log_question(question, status, sql, row_count, int((time.time() - start) * 1000))
        except Exception:
            pass  # a failed log write should not hide the answer
        st.session_state.asked += 1
    return record


def show_record(r):
    # One bordered card per question: the question as the card's heading, Genie's answer below it
    with st.container(border=True):
        st.markdown(f"**{r['question']}**")
        if r["text"]:
            st.write(r["text"])
        if r["error"]:
            st.error(r["error"])
        if r["df"] is not None:
            if r["df"].empty:
                st.info("The query ran but returned no rows.")
            else:
                chart = auto_chart(r["df"])
                if chart is not None:
                    st.altair_chart(chart, width="stretch")
                st.dataframe(r["df"], hide_index=True, width="stretch")
        if r["sql"]:
            with st.expander("SQL generated by Genie"):
                st.code(r["sql"], language="sql")
        for s in r["suggestions"]:
            st.button(s, key=f"sugg-{uuid.uuid4()}", on_click=queue_question, args=(s,))


def queue_question(q):
    st.session_state.pending = q


# Theme comes from .streamlit/config.toml at the repo root (Daniel Livingston brand: navy, teal, one amber highlight)
st.set_page_config(page_title="Ask the Fleet", page_icon=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "assets", "favicon.png"), layout="wide")
for key, default in [("session_id", str(uuid.uuid4())), ("asked", 0), ("history", []), ("pending", None)]:
    st.session_state.setdefault(key, default)

st.title("Ask the Fleet")
# The brand's short amber rule under the headline
st.markdown('<div style="width:56px;height:4px;background:#f0a742;border-radius:2px;margin:-0.5rem 0 1rem"></div>',
            unsafe_allow_html=True)
st.markdown(
    "Ask questions about 709 NASA turbofan engines that were run until they failed. "
    "Your question goes to a **Databricks Genie** space, which writes SQL against governed "
    "**Unity Catalog metric views** of reliability metrics; the answer comes straight from Databricks. "
    f"[Source code]({REPO_URL}) · [Guided tour of how it works]({TOUR_URL})"
)

ask_tab, report_tab, model_tab, how_tab = st.tabs(["Ask a question", "Fleet report", "Modeling", "How it works"])

with ask_tab:
    connected = True
    try:
        used_today = questions_used_today()
    except Exception as e:
        connected, used_today = False, 0
        st.error("The demo can't reach Databricks right now, so questions are turned off. Please try again later.")
        with st.expander("Details for the app owner"):
            st.write(str(e) if isinstance(e, SetupError) else f"{type(e).__name__}: {str(e)[:200]}")
    left_today = max(0, DAILY_LIMIT - used_today)
    left_session = max(0, SESSION_LIMIT - st.session_state.asked)
    can_ask = connected and left_today > 0 and left_session > 0

    if connected:
        st.caption(f"{left_today} of {DAILY_LIMIT} questions left today across all visitors · "
                   f"{left_session} of {SESSION_LIMIT} left for you. Questions are logged to improve the demo.")
        if not can_ask:
            st.warning("The question limit has been reached. It resets at midnight UTC. "
                       "The Fleet report tab still works.")

    st.write("Try one:")
    cols = st.columns(3)
    for i, s in enumerate(SUGGESTIONS):
        cols[i % 3].button(s, key=f"suggest-{i}", on_click=queue_question, args=(s,),
                           disabled=not can_ask, width="stretch")

    for r in st.session_state.history:
        show_record(r)

    typed = st.chat_input("Ask about engine life, failures, sensors or health stages",
                          max_chars=MAX_QUESTION_CHARS, disabled=not can_ask)
    question = typed or st.session_state.pending
    st.session_state.pending = None
    if question and can_ask:
        with st.spinner("Asking Genie..."):
            record = answer(question.strip()[:MAX_QUESTION_CHARS])
        st.session_state.history.append(record)
        st.rerun()

with report_tab:
    try:
        st.caption(f"Every number comes from the metric views {METRIC_VIEWS}, saved "
                   f"{load_snapshot()['exported_at'][:10]}. The NASA data is fixed, so the snapshot does not go stale.")
        render_report(fetch_snapshot, backend="databricks")
    except Exception as e:
        st.error("The report could not be loaded.")
        with st.expander("Details for the app owner"):
            st.write(f"{type(e).__name__}: {str(e)[:200]}")

with model_tab:
    render_modeling()

with how_tab:
    st.markdown(f"""
**Metrics are defined once, in Unity Catalog metric views.** `fleet_engines` (one row per engine) and
`fleet_cycles` (one row per engine per cycle) hold the reliability metrics, such as mean time to failure,
with comments and synonyms that help Genie map a question to the right definition. Keeping the two grains
apart means engine counts never fan out to cycle rows.

**They are checked on every pipeline run.** The Databricks Job that builds them recomputes each key metric
with plain SQL on the tables and fails if any value differs.

**This app is locked down.** It signs in as a Databricks service principal that can only read the metric
views and write to the question log. Genie runs its SQL under that identity, so a question can't change
data. Questions are capped at {DAILY_LIMIT} a day.

The same pipeline was first built on Snowflake (semantic view and Cortex Analyst); both versions are in
the repo. [Source code and full write-up]({REPO_URL})
""")

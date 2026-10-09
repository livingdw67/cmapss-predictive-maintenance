# Ask the Fleet: public demo of the FLEET_RELIABILITY semantic layer.
# Hosted on Streamlit Community Cloud. Visitors ask questions in plain English;
# Cortex Analyst turns them into SQL against the semantic view, and the app runs
# that SQL as a read-only service user on a small, capped warehouse.
import os
import re
import sys
import time
import uuid

import altair as alt
import pandas as pd
import requests
import snowflake.connector
import streamlit as st
from cryptography.hazmat.primitives import serialization

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "streamlit_app"))
from report import SEMANTIC_VIEW, render_report  # noqa: E402

REPO_URL = "https://github.com/livingdw67/cmapss-predictive-maintenance"
TOUR_URL = "https://claude.ai/artifact/Ai1U6EospZRsTgpSU9TxBL#tour"
# Each question costs about 0.067 Snowflake credits, paid by the project owner
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
# Generated SQL must be a single read and must not call any AI or system function.
# The service role can only read the semantic layer's tables; this check also keeps
# a crafted question from spending credits on Cortex functions, which the account
# grants to PUBLIC.
FORBIDDEN_SQL = re.compile(r"\bCORTEX\b|\bAI_[A-Z_]+\s*\(|\bCOMPLETE\s*\(|\bSYSTEM\$|;\s*\S", re.IGNORECASE)
SQL_COMMENTS = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)


class SetupError(Exception):
    """A configuration problem the app owner can fix; the message never contains secret values."""


@st.cache_resource
def get_connection():
    # Each step fails with its own message so a misconfigured deploy says what to fix
    try:
        has_section = "snowflake" in st.secrets
    except Exception:  # no secrets configured at all
        has_section = False
    if not has_section:
        raise SetupError("The app's secrets have no [snowflake] section. Paste the secrets file "
                         "into the app's Settings > Secrets on Streamlit Community Cloud.")
    cfg = st.secrets["snowflake"]
    missing = [k for k in ("account", "user", "role", "warehouse", "private_key") if not cfg.get(k)]
    if missing:
        raise SetupError(f"The [snowflake] secrets are missing: {', '.join(missing)}.")
    try:
        key = serialization.load_pem_private_key(cfg["private_key"].strip().encode(), password=None)
    except Exception as e:
        raise SetupError("The private_key secret could not be read. Paste it whole, including the "
                         f"BEGIN and END lines, inside triple quotes. ({type(e).__name__})")
    der = key.private_bytes(serialization.Encoding.DER, serialization.PrivateFormat.PKCS8,
                            serialization.NoEncryption())
    try:
        return snowflake.connector.connect(
            account=cfg["account"], user=cfg["user"], private_key=der,
            role=cfg["role"], warehouse=cfg["warehouse"],
            client_session_keep_alive=True, login_timeout=30,
        )
    except Exception as e:
        raise SetupError(f"Snowflake rejected the login: {str(e)[:200]}")


def fetch(sql, max_rows=None):
    cur = get_connection().cursor()
    cur.execute(sql)
    rows = cur.fetchmany(max_rows) if max_rows else cur.fetchall()
    return pd.DataFrame(rows, columns=[c[0] for c in cur.description])


def questions_used_today():
    return int(fetch("SELECT COUNT(*) FROM PREDICTIVE_MAINTENANCE.DEMO.QUESTION_LOG "
                     "WHERE asked_at >= CURRENT_DATE()").iloc[0, 0])


def log_question(question, status, sql, row_count, latency_ms):
    get_connection().cursor().execute(
        "INSERT INTO PREDICTIVE_MAINTENANCE.DEMO.QUESTION_LOG "
        "(session_id, question, status, generated_sql, row_count, latency_ms) VALUES (%s, %s, %s, %s, %s, %s)",
        (st.session_state.session_id, question, status, sql, row_count, latency_ms),
    )


def ask_analyst(question):
    """Send one question to Cortex Analyst; reconnect once if the session token expired."""
    for attempt in range(2):
        conn = get_connection()
        resp = requests.post(
            f"https://{conn.host}/api/v2/cortex/analyst/message",
            headers={"Authorization": f'Snowflake Token="{conn.rest.token}"',
                     "Content-Type": "application/json", "Accept": "application/json"},
            json={"messages": [{"role": "user", "content": [{"type": "text", "text": question}]}],
                  "semantic_view": SEMANTIC_VIEW},
            timeout=90,
        )
        if resp.status_code == 401 and attempt == 0:
            get_connection.clear()
            continue
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def clean_sql(sql):
    """Strip comments (Cortex Analyst appends one) and any trailing semicolon."""
    return SQL_COMMENTS.sub("", sql).strip().rstrip(";").strip()


def safe_to_run(sql):
    sql = clean_sql(sql)
    return sql.upper().startswith(("SELECT", "WITH")) and not FORBIDDEN_SQL.search(sql)


def auto_chart(df):
    # Bar chart when the answer is one label column and one number column
    if len(df) < 2 or len(df) > 40 or df.shape[1] != 2:
        return None
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric) != 1:
        return None
    value = numeric[0]
    label = next(c for c in df.columns if c != value)
    return alt.Chart(df).mark_bar(color="#2a78d6", cornerRadiusEnd=3).encode(
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
        content = ask_analyst(question)
        record["text"] = " ".join(c["text"] for c in content if c["type"] == "text")
        record["suggestions"] = next((c["suggestions"] for c in content if c["type"] == "suggestions"), [])
        sql = next((c["statement"] for c in content if c["type"] == "sql"), None)
        if sql is None:
            status = "no_sql"
        elif not safe_to_run(sql):
            status, record["error"] = "blocked", "This query was not run: the demo only runs read-only queries on the fleet data."
        else:
            sql = clean_sql(sql)
            record["sql"] = sql
            numeric = {}
            df = fetch(sql, MAX_ROWS)
            for col in df.columns:
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().all() and len(df):
                    numeric[col] = converted
            df = df.assign(**numeric)
            record["df"] = df
            status, row_count = "ok", len(df)
    except Exception as e:
        record["error"] = "Something went wrong answering that question. Try rephrasing it."
        record["detail"] = str(e)[:300]
    finally:
        log_question(question, status, sql, row_count, int((time.time() - start) * 1000))
        st.session_state.asked += 1
    return record


def show_record(r):
    with st.chat_message("user"):
        st.write(r["question"])
    with st.chat_message("assistant"):
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
            with st.expander("SQL generated by Cortex Analyst"):
                st.code(r["sql"], language="sql")
        for s in r["suggestions"]:
            st.button(s, key=f"sugg-{uuid.uuid4()}", on_click=queue_question, args=(s,))


def queue_question(q):
    st.session_state.pending = q


st.set_page_config(page_title="Ask the Fleet", page_icon="✈️", layout="wide")
for key, default in [("session_id", str(uuid.uuid4())), ("asked", 0), ("history", []), ("pending", None)]:
    st.session_state.setdefault(key, default)

st.title("Ask the Fleet")
st.markdown(
    "Ask questions about 709 NASA turbofan engines that were run until they failed. "
    "Your question goes to **Snowflake Cortex Analyst**, which writes SQL against a governed "
    "**semantic view** of reliability metrics; the answer comes straight from Snowflake. "
    f"[Source code]({REPO_URL}) · [Guided tour of how it works]({TOUR_URL})"
)

ask_tab, report_tab, how_tab = st.tabs(["Ask a question", "Fleet report", "How it works"])

with ask_tab:
    connected = True
    try:
        used_today = questions_used_today()
    except Exception as e:
        connected, used_today = False, 0
        st.error("The demo can't reach Snowflake right now, so questions are turned off. Please try again later.")
        with st.expander("Details for the app owner"):
            st.write(str(e) if isinstance(e, SetupError) else f"{type(e).__name__}: {str(e)[:200]}")
    left_today = max(0, DAILY_LIMIT - used_today)
    left_session = max(0, SESSION_LIMIT - st.session_state.asked)
    can_ask = connected and left_today > 0 and left_session > 0

    if connected:
        st.caption(f"{left_today} of {DAILY_LIMIT} questions left today across all visitors · "
                   f"{left_session} of {SESSION_LIMIT} left for you. Questions are logged to improve the demo.")
        if not can_ask:
            st.warning("The question limit has been reached. It resets at midnight Pacific time. "
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
        with st.spinner("Asking Cortex Analyst..."):
            record = answer(question.strip()[:MAX_QUESTION_CHARS])
        st.session_state.history.append(record)
        st.rerun()

with report_tab:
    st.caption(f"Every number comes from the semantic view `{SEMANTIC_VIEW}`. Filters cost a fraction of a cent.")
    try:
        render_report(fetch)
    except Exception:
        st.error("The report can't reach Snowflake right now. Please try again later.")

with how_tab:
    st.markdown(f"""
**The semantic view is the only place metrics are defined.** `FLEET_RELIABILITY` models three tables at three
grains (sub-fleets, engines, cycles) with 16 metrics such as mean time to failure, plus synonyms and
instructions that help Cortex Analyst map a question to the right definition.

**It is checked two ways.** Every metric is recomputed with separate SQL on the raw staging data (80 of 80
values match), and an evaluation asks 25 questions three times each and scores the returned answers (75 of 75).

**This app is locked down.** It connects as a service user that can only log in with a key, runs as a
read-only role on its own extra-small warehouse with a 30-second query limit and a monthly credit cap, and
only runs SQL that reads the semantic view. Questions are capped at {DAILY_LIMIT} a day.

[Source code and full write-up]({REPO_URL})
""")

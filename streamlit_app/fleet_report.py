# Fleet reliability report (Streamlit in Snowflake).
# The page itself lives in report.py, shared with the public app; this file only
# supplies the Snowpark session that Streamlit in Snowflake provides.
import streamlit as st
from snowflake.snowpark.context import get_active_session

from report import SEMANTIC_VIEW, render_report

st.set_page_config(page_title="Fleet Reliability", layout="wide")
st.title("Fleet reliability")
st.caption("NASA CMAPSS turbofan fleet, all engines run to failure. "
           f"Source: semantic view `{SEMANTIC_VIEW}`.")

render_report(lambda sql: get_active_session().sql(sql).to_pandas())

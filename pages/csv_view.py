import os
from pathlib import Path
import pandas as pd
import streamlit as st

st.title("CSV Viewer")

DATA_DIR = "data"

# get the latest csv from the download die
csv = sorted(filter(lambda p: p.endswith(".csv"), os.listdir(DATA_DIR)))[-1]

if not csv:
    st.info("Run download to view bulk historic data.")
    
else:
    df = pd.read_csv(Path(DATA_DIR) / csv)

    st.dataframe(df, use_container_width=False, height=700)


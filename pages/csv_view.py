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

    selected_sectors = st.multiselect("Sectors", df["gics_sector"].unique())
    selected_types = st.multiselect("Types", df["asx_type_label"].unique())
    max_n_obs = int(df["early_n_observations"].max())
    min_obs, max_obs = st.slider("Num Observations Range", 0, max_n_obs, (0, max_n_obs))

    # filter by sector
    if selected_sectors:
        df = df[df["gics_sector"].isin(selected_sectors)]

    if selected_types:
        df = df[df["asx_type_label"].isin(selected_types)]

    df = df[df["early_n_observations"].between(min_obs, max_obs)]

    st.dataframe(df, use_container_width=False, height=700)

"""Streamlit app for visualizing predictions of trained models."""


import streamlit as st

import app_utils


st.set_page_config(page_title="Gestura", page_icon="🤌", layout="wide")

if not st.session_state.get("init"):
    app_utils.first_time()

st.title("Gestura")
st.write("Gestura is an app for visualizing & introspecting predictions of gestures from time-series data.")
st.write("Explore the other pages in the sidebar for more details.")

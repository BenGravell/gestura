"""Streamlit app for visualizing predictions of trained models."""


import streamlit as st

import app_utils


st.set_page_config(page_title="Gestura", page_icon="🤌", layout="wide")

if not st.session_state.get("init"):
    app_utils.first_time()

st.title("Gestura")
st.subheader("*An app for visualizing & introspecting predictions of gestures from time-series data.*", anchor=False)
st.divider()
st.subheader("Use Cases", anchor=False)
st.markdown("""
- Get a quantitative assessment of the model's predictive capabilities.
- Introspect individual examples to understand *why* the model arrived at its predictions.
- Learn about state-of-the-art deep learning model architectures.
- Learn about open-source time-series classification datasets.
""")

st.success("Explore the other pages in the sidebar for more details.", icon="↩️")

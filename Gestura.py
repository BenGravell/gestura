"""Streamlit app for visualizing predictions of trained models."""


import streamlit as st

import app_utils


def streamlit_setup():
    st.set_page_config(page_title="Gestura", page_icon="🤌", layout="wide")


def welcome():
    st.title("🤌 Gestura")
    st.caption("Visualize & introspect predictions of gestures from time-series data.")
    
    cols = st.columns([2, 1])

    with cols[0]:
        st.subheader("What is Gestura?", divider="blue", anchor=False)

        st.markdown("Gestura is an app that lets you:")
        st.markdown("""
        - Get a quantitative assessment of the model's predictive capabilities.
        - Introspect individual examples to understand *why* the model arrived at its predictions.
        - Learn about state-of-the-art deep learning model architectures.
        - Learn about open-source time-series classification datasets.
        """)

        st.success("Explore the other pages in the sidebar for more details.", icon="↩️")

    with cols[1]:
        st.image("assets/welcome.jpg")


if __name__ == "__main__":
    streamlit_setup()    

    if not st.session_state.get("init"):
        app_utils.first_time()

    welcome()
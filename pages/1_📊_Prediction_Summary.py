import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards

import config
import utils
import app_utils


st.set_page_config(page_title="Gestura", page_icon="🤌", layout="wide")

style_metric_cards(
    border_left_color=config.STREAMLIT_CONFIG["theme"]["primaryColor"],
    border_color=config.STREAMLIT_CONFIG["theme"]["secondaryBackgroundColor"],
    background_color=config.STREAMLIT_CONFIG["theme"]["backgroundColor"],
    border_size_px=2,
    border_radius_px=20,
    box_shadow=False,
)

if not st.session_state.get("init"):
    app_utils.first_time()

app_utils.show_load_most_recent_model_button_in_sidebar()

# Prepare data
gesture_df = app_utils.get_gesture_df()
df = app_utils.get_predictions(st.session_state.model_path)
confusion_matrix = app_utils.get_confusion_matrix(st.session_state.model_path)
metrics = app_utils.compute_metrics(df["ground_truth"], df["predicted"])

# Prepare data for display
metrics_data = {
    "Class": np.unique(df["ground_truth"]),
    "Gesture": gesture_df["gesture"].values,
    "Precision": 100 * metrics["per_class"]["precision"],
    "Recall": 100 * metrics["per_class"]["recall"],
    "F1-Score": 100 * metrics["per_class"]["f1"],
}
metrics_df = pd.DataFrame(metrics_data).set_index("Class")


st.title("Prediction Summary")
st.caption(
    "Summary of predictive performance on the test set. None of the examples shown here have been exposed to the model during training."
)

st.header("Prediction Table", anchor=False, divider="blue")
prediction_table_df = df.copy()
prediction_table_df = prediction_table_df.rename_axis("Example Index")
prediction_table_df["predicted"] = prediction_table_df["predicted"].apply(app_utils.class_index_to_gesture_image_url)
prediction_table_df["ground_truth"] = prediction_table_df["ground_truth"].apply(app_utils.class_index_to_gesture_image_url)
prediction_table_df = prediction_table_df.rename(
    columns={"predicted": "Prediction", "ground_truth": "Ground Truth", "correct": "Correct"}
)
st.dataframe(
    prediction_table_df,
    column_config={
        "Prediction": st.column_config.ImageColumn("Prediction", width="small", help="The intended gesture."),
        "Ground Truth": st.column_config.ImageColumn("Ground Truth", width="small", help="The intended gesture."),
    },
    use_container_width=True,
)

st.header("Prediction Quality Metrics", anchor=False, divider="blue")


st.subheader(
    "Overall Metrics",
    help=(
        "Precision and recall use the 'weighted' average method; see the [scikit-learn docs for"
        " precision_score()](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)."
    ),
    anchor=False,
)
metric_cols = st.columns(3)
metric_cols[0].metric("Accuracy", f'{app_utils.pct_fmt(metrics["overall"]["accuracy"])}')
metric_cols[1].metric("Precision", f'{app_utils.pct_fmt(metrics["overall"]["precision"])}')
metric_cols[2].metric("Recall", f'{app_utils.pct_fmt(metrics["overall"]["recall"])}')

st.subheader(
    "Per-Class Metrics",
    help=(
        "See the [scikit-learn docs for"
        " precision_recall_fscore_support()](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html)."
    ),
    anchor=False,
)

image_cc = st.column_config.ImageColumn(width="small")

percentage_cc = st.column_config.NumberColumn(
    min_value=0,
    max_value=100,
    step=1,
    format="%d%%",
)

st.dataframe(
    metrics_df,
    column_config={
        "Precision": percentage_cc,
        "Recall": percentage_cc,
        "F1-Score": percentage_cc,
        "Gesture": image_cc,
    },
    use_container_width=True,
)

st.header("Confusion Matrix", anchor=False, divider="blue")
x = [f"Predicted Class {i}" for i in range(utils.output_size)]
y = [f"Actual Class {i}" for i in range(utils.output_size)]

fig_cm = ff.create_annotated_heatmap(
    z=confusion_matrix[::-1], x=x, y=y[::-1], colorscale="Blues_r", showscale=True, reversescale=True
)

st.plotly_chart(fig_cm, use_container_width=True)
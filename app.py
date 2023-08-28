"""Streamlit app for visualizing predictions of trained models."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchviz import make_dot
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_fscore_support

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

import utils


st.set_page_config(layout="wide")


@st.cache_data
def load_data(dataset_name="UWaveGestureLibrary"):
    return utils.load_dataset(dataset_name)


@st.cache_resource(max_entries=10)
def load_model(model_path):
    model = utils.load_checkpoint(model_path)["model"]
    model.eval()
    return model


_, _, dataset_test, _ = load_data()
dataloader = DataLoader(dataset_test, batch_size=64, shuffle=False)


def load_most_recent_model_callback():
    most_recent_model_path = utils.get_most_recent_model_path()
    if most_recent_model_path == st.session_state.get("model_path"):
        st.toast(f"Model already loaded from :green[`{st.session_state.model_path}`]", icon="✅")
    else:
        st.session_state.model_path = utils.get_most_recent_model_path()
        st.session_state.model = load_model(st.session_state.model_path)
        st.toast(f"Loaded model from :green[`{st.session_state.model_path}`]", icon="🔄")


def first_time():
    load_most_recent_model_callback()
    st.session_state.init = True


if not st.session_state.get("init"):
    first_time()


st.button(
    "Load Most Recent Model",
    on_click=load_most_recent_model_callback,
    type="primary",
    help=(
        "Load the model with the most recent timestamp. Use this to inspect predictions as the model is training in"
        " realtime."
    ),
)


tab_names = ["Prediction Summary", "Example Inspector", "Model Information", "Dataset Information"]
tabs = st.tabs(tab_names)


with tabs[tab_names.index("Dataset Information")]:
    st.subheader("Convenient Source", anchor=False)
    url = "http://www.timeseriesclassification.com/description.php?Dataset=UWaveGestureLibrary"
    st.write(
        f"The dataset used here is the [UWaveGestureLibrary]({url})."
    )
    # NOTE: the iframe embedded url works locally but not when hosted on Streamlit Cloud.
    # st.components.v1.iframe(url, height=600, scrolling=True)

    st.subheader("Description", anchor=False)
    st.write('A set of eight simple gestures generated from accelerometers. The data consists of the X, Y, Z coordinates of each motion. Each series has a length of 315.')

    st.subheader("Original Source", anchor=False)
    st.write('The dataset was introduced by J. Liu, Z. Wang, L. Zhong, J. Wickramasuriya and V. Vasudevan, in "uWave: Accelerometer-based personalized gesture recognition and its applications," 2009 IEEE International Conference on Pervasive Computing and Communications, Galveston, TX, 2009, pp. 1-9.')
    st.write("https://ieeexplore.ieee.org/document/4912759")

    st.subheader("Label Definitions", anchor=False)
    st.image("http://www.timeseriesclassification.com/images/datasets/UWaveGestureLibrary.jpg")

    st.subheader("Download Link", anchor=False)
    st.write("http://www.timeseriesclassification.com/aeon-toolkit/UWaveGestureLibrary.zip")


@st.cache_resource(max_entries=1)
def gen_diagram():
    model = utils.LSTMWithAttention()
    x = torch.zeros(1, utils.sequence_length, utils.input_size)
    digraph = make_dot(model(x), params=dict(model.named_parameters()))
    return digraph


with tabs[tab_names.index("Model Information")]:
    st.write("The model used for prediction is a Long Short-Term Memory (LSTM) neural network with multi-head scaled dot-product self-attention.")
    st.write("The architecture of the model is shown by the graph below.")
    st.graphviz_chart(gen_diagram(), use_container_width=True)


@st.cache_data
def get_predictions(model_path):
    # Crude check to ensure that the model in st.session_state.model corresponds to model_path
    # This is to enable caching based on model_path rather than the model itself
    assert model_path == st.session_state.model_path

    all_ground_truth = []
    all_predictions = []

    with torch.no_grad():
        for feature, label in dataloader:
            outputs = st.session_state.model(feature)

            # Use argmax to get class predictions if your outputs are probabilities
            predicted_classes = torch.argmax(outputs, dim=1)

            all_predictions.extend(predicted_classes.detach().numpy())
            all_ground_truth.extend(label.detach().numpy())

    df = pd.DataFrame({"ground_truth": all_ground_truth, "predicted": all_predictions})
    df["correct"] = df["ground_truth"] == df["predicted"]
    return df


df = get_predictions(st.session_state.model_path)


def compute_metrics(y_true, y_pred):
    # Compute overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")

    # Compute per-class precision, recall, and F1-score
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.unique(y_true)
    )

    return {
        "overall": {"accuracy": accuracy, "precision": precision, "recall": recall},
        "per_class": {"precision": precision_per_class, "recall": recall_per_class, "f1": f1_per_class},
    }


@st.cache_data
def get_confusion_matrix(model_path):
    # Crude check to ensure that the model in st.session_state.model corresponds to model_path
    # This is to enable caching based on model_path rather than the model itself
    assert model_path == st.session_state.model_path

    confusion_matrix = np.zeros((utils.output_size, utils.output_size), int)
    for idx, row in df.iterrows():
        confusion_matrix[row.ground_truth, row.predicted] += 1
    return confusion_matrix


confusion_matrix = get_confusion_matrix(st.session_state.model_path)


def class_index_to_gesture_image_url(i):
    return f"https://raw.githubusercontent.com/BenGravell/gestura/main/gesture_images/gesture_{i}.png"


gesture = pd.DataFrame.from_dict(
    {i: {"gesture": class_index_to_gesture_image_url(i)} for i in range(utils.NUM_CLASSES)}, orient="index"
)


def pct_fmt(x):
    return f"{round(100*x)}%"


image_cc = st.column_config.ImageColumn(width="small")

percentage_cc = st.column_config.NumberColumn(
    min_value=0,
    max_value=100,
    step=1,
    format="%d%%",
)

with tabs[tab_names.index("Prediction Summary")]:
    st.write("The Prediction Summary tab shows predictive performance on the test set. None of the examples shown here have been exposed to the model during training.")
    cols = st.columns(3)

    with cols[0]:
        st.subheader("Prediction Table", anchor=False)
        prediction_table_df = df.copy()
        prediction_table_df = prediction_table_df.rename_axis("Example Index")
        prediction_table_df["predicted"] = prediction_table_df["predicted"].apply(class_index_to_gesture_image_url)
        prediction_table_df["ground_truth"] = prediction_table_df["ground_truth"].apply(
            class_index_to_gesture_image_url
        )
        prediction_table_df = prediction_table_df.rename(
            columns={"predicted": "Prediction", "ground_truth": "Ground Truth", "correct": "Correct"}
        )
        st.dataframe(
            prediction_table_df,
            column_config={
                "Prediction": st.column_config.ImageColumn("Prediction", width="small", help="The intended gesture."),
                "Ground Truth": st.column_config.ImageColumn(
                    "Ground Truth", width="small", help="The intended gesture."
                ),
            },
            use_container_width=True,
        )

    with cols[1]:
        st.subheader("Prediction Quality Metrics", anchor=False)
        metrics = compute_metrics(df["ground_truth"], df["predicted"])

        # Prepare data for display
        metrics_data = {
            "Class": np.unique(df["ground_truth"]),
            "Gesture": gesture["gesture"].values,
            "Precision": 100 * metrics["per_class"]["precision"],
            "Recall": 100 * metrics["per_class"]["recall"],
            "F1-Score": 100 * metrics["per_class"]["f1"],
        }

        metrics_df = pd.DataFrame(metrics_data).set_index("Class")

        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric("Accuracy", f'{pct_fmt(metrics["overall"]["accuracy"])}')
        with metric_cols[1]:
            st.metric("Precision", f'{pct_fmt(metrics["overall"]["precision"])}')
        with metric_cols[2]:
            st.metric("Recall", f'{pct_fmt(metrics["overall"]["recall"])}')

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

    with cols[2]:
        st.subheader("Confusion Matrix", anchor=False)

        x = [f"Predicted Class {i}" for i in range(utils.output_size)]
        y = [f"Actual Class {i}" for i in range(utils.output_size)]

        fig_cm = ff.create_annotated_heatmap(
            z=confusion_matrix[::-1], x=x, y=y[::-1], colorscale="matter", showscale=True, reversescale=True
        )

        st.plotly_chart(fig_cm, use_container_width=True)

with tabs[tab_names.index("Example Inspector")]:
    st.write("The Example Inspector tab shows predictions and details for individual examples in the test set. None of the examples shown here have been exposed to the model during training.")
    idx = st.number_input("Example Index", min_value=0, max_value=len(dataset_test) - 1)
    feature, label = dataset_test[idx]

    feature_df = pd.DataFrame(feature)
    feature_dim_names = ["x", "y", "z"]
    feature_df.columns = feature_dim_names

    x = feature[None, :]
    output, attn = st.session_state.model.forward_with_attn(x)

    predicted_label = torch.argmax(output, dim=1).numpy().astype(int).item()
    label = label.numpy().astype(int).item()

    labels = np.arange(utils.NUM_CLASSES).astype(float)
    predicted = output.detach().numpy()[0]

    cols = st.columns((2, 4, 4, 4))
    with cols[0]:
        im_width = 125
        st.subheader("Ground Truth Gesture", anchor=False)
        st.image(f"gesture_images/gesture_{label}.png", width=im_width)
        st.subheader("Predicted Gesture", anchor=False)
        st.image(f"gesture_images/gesture_{predicted_label}.png", width=im_width)
    with cols[1]:
        st.subheader("3D Trajectory", anchor=False)
        hover_text = [f"Timestep: {i:3d}" for i in range(len(feature_df))]

        trace = go.Scatter3d(
            x=feature_df["x"],
            y=feature_df["y"],
            z=feature_df["z"],
            text=hover_text,
            hoverinfo="text+x+y+z",
            marker=dict(
                size=3,
                color="#67B3FB",
            ),
            line=dict(color="white", width=3),
        )

        data = [trace]

        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))

        fig = go.Figure(data=data, layout=layout)
        st.plotly_chart(fig, use_container_width=True)
    with cols[2]:
        st.subheader("Time Series & Attention", anchor=False)

        plot_container = st.container()
        options_container = st.container()

        with options_container:
            head_idx = st.number_input("Attention Head Index", min_value=0, max_value=utils.heads - 1)
            predicted_attn = attn[0, head_idx].detach().numpy()
        with plot_container:
            line = px.line(feature_df)

            # Create a subplot with 2 rows and 1 column
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                subplot_titles=("Time Series Data", f"Attention Heatmap (Head {head_idx})"),
                row_heights=[0.4, 0.6],  # Giving more height to the heatmap
            )
            # Add line plot for time series data
            fig.add_traces(list(line.select_traces()), rows=1, cols=1)

            # Add heatmap for attention weights
            fig.add_trace(
                go.Heatmap(
                    z=predicted_attn,
                    colorscale="matter_r",
                    showscale=True,
                    colorbar=dict(y=0.2, len=0.5),  # Adjust position and length of the color scale
                ),
                row=2,
                col=1,
            )

            st.plotly_chart(fig, use_container_width=True)

    with cols[3]:
        st.subheader("Label Prediction", anchor=False)
        fig = px.bar(x=labels, y=predicted)



        # Add lines and annotations for the prediction and ground truth
        fig.add_shape(
            type="line",
            x0=label - 0.1,
            x1=label - 0.1,
            y0=-10,
            y1=10,
            line=dict(color="white", width=2, dash="dash"),
            name="Ground Truth",
        )
        fig.add_shape(
            type="line",
            x0=predicted_label + 0.1,
            x1=predicted_label + 0.1,
            y0=-10,
            y1=10,
            line=dict(color="orange", width=2, dash="dash"),
            name="Prediction",
        )

        fig.add_annotation(
            text="Ground Truth",
            x=label - 0.1 - 0.2,
            y=-6,
            arrowhead=2,
            showarrow=False,
            font=dict(size=14, color="white"),
            textangle=-90,  # Rotate the text by -90 degrees
        )
        fig.add_annotation(
            text="Prediction",
            x=predicted_label + 0.1 + 0.2,
            y=-6,
            arrowhead=2,
            showarrow=False,
            font=dict(size=14, color="orange"),
            textangle=-90,  # Rotate the text by -90 degrees
        )
        # Customize the layout
        fig.update_layout(
            xaxis_title="Class",
            yaxis_title="Value",
            xaxis=dict(
                tickvals=list(range(0, utils.NUM_CLASSES)),
                ticktext=[str(i) for i in range(0, utils.NUM_CLASSES)]
            )
        )

        st.plotly_chart(fig, use_container_width=True)

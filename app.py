"""Streamlit app for visualizing predictions of trained models."""

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torchviz import make_dot
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_fscore_support

import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

import config
import utils


st.set_page_config(page_title="Gestura", page_icon="🤌", layout="wide")

style_metric_cards(
    border_left_color=config.STREAMLIT_CONFIG["theme"]["primaryColor"],
    border_color=config.STREAMLIT_CONFIG["theme"]["secondaryBackgroundColor"],
    background_color=config.STREAMLIT_CONFIG["theme"]["backgroundColor"],
    border_size_px=2,
    border_radius_px=20,
    box_shadow=False,
)

# Define a container before the tabs for topline options like the Load Most Recent Model button
topline_container = st.container()

# NOTE: It is critical to define the tabs first before any other operations that
# conditionally add content to the main body of the app to avoid a
# jump-to-first-tab-on-first-interaction bug
tab_names = ["Prediction Summary", "Example Inspector", "Model Information", "Dataset Information"]
tabs = st.tabs(tab_names)


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

with topline_container:
    st.button(
        "Load Most Recent Model",
        on_click=load_most_recent_model_callback,
        type="primary",
        help=(
            "Load the model with the most recent timestamp. Use this to inspect predictions as the model is training in"
            " realtime."
        ),
    )


def class_index_to_gesture_image_url(i, local=False):
    path = f"gesture_images/gesture_{i}.png"
    if not local:
        path = f"https://raw.githubusercontent.com/BenGravell/gestura/main/{path}"
    return path


with tabs[tab_names.index("Dataset Information")]:
    st.header("Data Description", anchor=False)

    st.write("The data represent accelerometer recordings of human users making one of of eight simple gestures.")

    st.subheader("Features", anchor=False)
    st.write(
        "The features consist of the acceleration in the X, Y, Z directions for each motion, with each series having a"
        " length of 315. The data were captured using an accelerometer anchored to the user's hand."
    )

    st.subheader("Labels", anchor=False)
    st.write("The labels consist of the ID associated with the intended gesture made by the user.")
    gesture_cols = st.columns(utils.NUM_CLASSES)
    for i in range(utils.NUM_CLASSES):
        with gesture_cols[i]:
            st.image(class_index_to_gesture_image_url(i, local=True), caption=f"Gesture {i}")
    st.write(
        'Gesture vocabulary adopted from [C.S. Myers, L.R. Rabiner, *"A comparative study of several dynamic'
        ' time-warping algorithms for connected word recognition,"* The Bell System Technical Journal 60 (1981)'
        " 1389-1409](https://ieeexplore.ieee.org/document/6771178). The dot denotes the start, and the arrow denotes"
        " the end."
    )

    st.header("Source", anchor=False)
    url = "http://www.timeseriesclassification.com/description.php?Dataset=UWaveGestureLibrary"
    st.write(
        f"The dataset used here is the [UWaveGestureLibrary]({url}) as provided by"
        " [timeseriesclassification.com](https://www.timeseriesclassification.com/) and the [aeon"
        " toolkit](https://www.aeon-toolkit.org/)."
    )

    st.write(
        'The dataset was introduced by [J. Liu, Z. Wang, L. Zhong, J. Wickramasuriya and V. Vasudevan, *"uWave:'
        ' Accelerometer-based personalized gesture recognition and its applications,"* 2009 IEEE International'
        " Conference on Pervasive Computing and Communications, Galveston, TX, 2009, pp."
        " 1-9](https://ieeexplore.ieee.org/document/4912759)."
    )

    st.header("Download Link", anchor=False)
    st.write("http://www.timeseriesclassification.com/aeon-toolkit/UWaveGestureLibrary.zip")


@st.cache_resource(max_entries=1)
def gen_diagram():
    model = utils.LSTMWithAttention()
    x = torch.zeros(1, utils.sequence_length, utils.input_size)
    digraph = make_dot(model(x), params=dict(model.named_parameters()))
    return digraph


def expander_markdown_from_file(title, path):
    with open(path, "r") as file:
        markdown_content = file.read()
    with st.expander(title):
        st.markdown(markdown_content)


with tabs[tab_names.index("Model Information")]:
    st.header("Model Architecture Summary", anchor=False, divider="blue")
    st.write(
        "The model used for prediction is a Long Short-Term Memory (LSTM) neural network with multi-head scaled"
        " dot-product self-attention."
    )

    st.subheader("Long Short-Term Memory (LSTM)", anchor=False)
    st.write(
        "LSTMs are a type of recurrent neural network (RNN) architecture. While vanilla RNNs can theoretically capture"
        " long-range dependencies in sequential data, they often struggle to do so in practice due to the vanishing and"
        " exploding gradient problems. LSTMs were designed to address these issues."
    )
    expander_markdown_from_file("Key features of LSTMs", "help/lstm_key_features.md")
    expander_markdown_from_file("Handy References for LSTMs", "help/lstm_handy_references.md")

    st.subheader("Self-Attention", anchor=False)
    st.write(
        "Self-attention, especially as popularized by the Transformer architecture, is a mechanism that enables a"
        " neural network to focus on different parts of the input data relative to a particular position in the data,"
        " often in the context of sequences. "
    )
    expander_markdown_from_file("Key features of Self-Attention", "help/self_attention_key_features.md")
    expander_markdown_from_file("Handy References for Self-Attention", "help/self_attention_handy_references.md")

    st.subheader("Putting Recurrence & Self-Attention Together", anchor=False)
    st.write(
        "The combination of self-attention and recurrent neural networks has been shown to be be more effective than"
        " either self-attention or recurrence individually in time-series classification tasks by [Katrompas,"
        ' Alexander, Theodoros Ntakouris, and Vangelis Metsis. *"Recurrence and self-attention vs the transformer for'
        ' time-series classification: a comparative study."* International Conference on Artificial Intelligence in'
        " Medicine. Cham: Springer International Publishing,"
        " 2022](https://link.springer.com/chapter/10.1007/978-3-031-09342-5_10)."
    )

    st.header("Model Architecture Details", anchor=False, divider="blue")
    st.write("The full architecture of the model is shown by the graph below.")
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
    st.write(
        "The *Prediction Summary* tab shows predictive performance on the test set. None of the examples shown here"
        " have been exposed to the model during training."
    )

    st.header("Prediction Table", anchor=False, divider="blue")
    prediction_table_df = df.copy()
    prediction_table_df = prediction_table_df.rename_axis("Example Index")
    prediction_table_df["predicted"] = prediction_table_df["predicted"].apply(class_index_to_gesture_image_url)
    prediction_table_df["ground_truth"] = prediction_table_df["ground_truth"].apply(class_index_to_gesture_image_url)
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

    st.subheader(
        "Overall Metrics",
        help=(
            "Precision and recall use the 'weighted' average method; see the [scikit-learn docs for"
            " precision_score()](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)."
        ),
        anchor=False,
    )
    metric_cols = st.columns(3)
    metric_cols[0].metric("Accuracy", f'{pct_fmt(metrics["overall"]["accuracy"])}')
    metric_cols[1].metric("Precision", f'{pct_fmt(metrics["overall"]["precision"])}')
    metric_cols[2].metric("Recall", f'{pct_fmt(metrics["overall"]["recall"])}')

    st.subheader(
        "Per-Class Metrics",
        help=(
            "See the [scikit-learn docs for"
            " precision_recall_fscore_support()](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html)."
        ),
        anchor=False,
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

with tabs[tab_names.index("Example Inspector")]:
    st.write(
        "The *Example Inspector* tab shows predictions and details for individual examples in the test set. None of the"
        " examples shown here have been exposed to the model during training."
    )
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
    predicted_proba = softmax(output, dim=1).detach().numpy()[0]

    cols = st.columns([2, 3])
    with cols[0]:
        st.header("3D Acceleration Trajectory", anchor=False, divider="blue")
        st.info(
            "This is **not** the literal 3D trajectory of *positions* in an inertial frame, but rather the 3D"
            " trajectory of *accelerations* as recorded by the accelerometer.",
            icon="📢",
        )
        hover_text = [f"Timestep: {i:3d}" for i in range(len(feature_df))]

        trace = go.Scatter3d(
            x=feature_df["x"],
            y=feature_df["y"],
            z=feature_df["z"],
            text=hover_text,
            hoverinfo="text+x+y+z",
            marker=dict(
                size=3,
                color=config.STREAMLIT_CONFIG["theme"]["primaryColor"],
            ),
            line=dict(color="black", width=3),
        )

        data = [trace]

        layout = go.Layout(height=500, margin=dict(l=0, r=0, b=0, t=0))

        fig = go.Figure(data=data, layout=layout)
        st.plotly_chart(fig, use_container_width=True)

    with cols[1]:
        st.header("Time Series & Attention", anchor=False, divider="blue")

        plot_container = st.container()
        options_container = st.container()

        with options_container:
            # head_idx = st.number_input("Attention Head Index", min_value=0, max_value=utils.heads - 1)
            head_idx = st.radio("Attention Head Index", options=[i for i in range(utils.heads)], horizontal=True)
            predicted_attn = attn[0, head_idx].detach().numpy()
        with plot_container:
            line = px.line(feature_df)

            # Create a subplot with 2 rows and 1 column
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                subplot_titles=("Time Series Data", f"Attention Heatmap (Head {head_idx})"),
                row_heights=[0.3, 0.7],  # Giving more height to the heatmap
                vertical_spacing=0.1,
            )
            # Add line plot for time series data
            fig.add_traces(list(line.select_traces()), rows=1, cols=1)

            # Add heatmap for attention weights
            fig.add_trace(
                go.Heatmap(
                    z=predicted_attn,
                    colorscale="Blues",
                    showscale=True,
                    colorbar=dict(y=0.3, len=0.6),  # Adjust position and length of the color scale
                ),
                row=2,
                col=1,
            )

            # fig.update_yaxes(scaleanchor="x", scaleratio=1, row=2, col=1)
            fig.update_layout(height=600, margin=dict(l=50, r=50, b=0, t=0))
            st.plotly_chart(fig, use_container_width=True)

    st.header("Predicted Class Probabilties", anchor=False, divider="blue")
    fig = px.bar(x=labels, y=predicted_proba)

    fig.add_annotation(
        text="Ground Truth",
        x=label,
        y=-0.2,
        arrowhead=2,
        showarrow=False,
        font=dict(size=14, color="black"),
        textangle=0,
        xanchor="center",
    )
    fig.add_annotation(
        text="Prediction",
        x=predicted_label,
        y=-0.1,
        arrowhead=2,
        showarrow=False,
        font=dict(size=14, color=config.STREAMLIT_CONFIG["theme"]["primaryColor"]),
        textangle=0,
        xanchor="center",
    )
    # Customize the layout
    fig.update_layout(
        xaxis_title="Class",
        yaxis_title="Value",
        xaxis=dict(tickvals=list(range(0, utils.NUM_CLASSES)), ticktext=[str(i) for i in range(0, utils.NUM_CLASSES)]),
    )

    # Adjust the y-axis range to leave space for the images
    fig.update_yaxes(range=[-0.8, 1.0])

    for i in range(utils.NUM_CLASSES):
        fig.add_layout_image(
            x=i,
            y=-0.5,
            source=class_index_to_gesture_image_url(i),
            xref="x",
            yref="y",
            sizex=0.4,
            sizey=0.4,
            xanchor="center",
            yanchor="middle",
        )

    st.plotly_chart(fig, use_container_width=True)

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torchviz import make_dot
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import streamlit as st

import utils


# CONSTANTS

labels = np.arange(utils.NUM_CLASSES)
label_names_map = {
    0: "Angle Down",
    1: "Square CW",
    2: "Straight Right",
    3: "Straight Left",
    4: "Straight Up",
    5: "Straight Down",
    6: "Circle CW",
    7: "Circle CCW",
}

head_cols = [f"Head {i}" for i in range(utils.heads)]


################################################################################
# Rendering utils
################################################################################


def get_colors_from_colormap(colormap_name, n, start=0, end=1):
    """Return n evenly spaced colors from a given colormap within a specified range."""
    colormap = plt.cm.get_cmap(colormap_name)

    # Generate n evenly spaced numbers between start and end
    values = np.linspace(start, end, n)
    colors = [colormap(value) for value in values]

    # Convert the RGBA colors to RGB format for use in Plotly
    rgb_colors = [f"rgb({int(255*color[0])}, {int(255*color[1])}, {int(255*color[2])})" for color in colors]
    return rgb_colors


def class_index_to_gesture_image_url(i, local=False):
    path = f"gesture_images/gesture_{i}.png"
    if not local:
        path = f"https://raw.githubusercontent.com/BenGravell/gestura/main/{path}"
    return path


def get_gesture_df():
    return pd.DataFrame.from_dict(
        {i: {"gesture": class_index_to_gesture_image_url(i)} for i in range(utils.NUM_CLASSES)}, orient="index"
    )


def expander_markdown_from_file(title, path):
    with open(path, "r") as file:
        markdown_content = file.read()
    with st.expander(title):
        st.markdown(markdown_content)


def pct_fmt(x):
    return f"{round(100*x)}%"


def show_load_model_button_in_sidebar():
    with st.sidebar:
        st.button(
            "Load Most Recent Model",
            on_click=load_model_callback,
            kwargs={"model_path": utils.get_most_recent_model_path()},
            type="primary",
            use_container_width=True,
        )
        st.caption(
            "Load the model that was saved most recently, as determined by timestamp in filename. Use this to inspect"
            " predictions as the model is training in realtime."
        )


################################################################################
# Data & model utils
################################################################################


@st.cache_data(max_entries=10)
def load_dataset(dataset_name=None):
    return utils.load_dataset(dataset_name)


@st.cache_data(max_entries=10)
def load_test_dataset_and_noshuffle_dataloader(dataset_name=None):
    _, _, dataset, _ = load_dataset(dataset_name)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    return dataset, dataloader


def load_dataset_callback(dataset_name=None):
    if dataset_name == st.session_state.get("dataset_name"):
        st.toast(f"Dataset :green[`{dataset_name}`] already loaded", icon="✅")
    else:
        dataset, dataloader = load_test_dataset_and_noshuffle_dataloader(dataset_name)
        st.session_state.dataset_name = dataset_name
        st.session_state.dataset = dataset
        st.session_state.dataloader = dataloader
        st.toast(f"Loaded dataset :green[`{dataset_name}`]", icon="🔄")


@st.cache_resource(max_entries=10)
def load_model(model_path):
    model = utils.load_checkpoint(model_path)["model"]
    model.eval()
    return model


def load_model_callback(model_path):
    if model_path == st.session_state.get("model_path"):
        st.toast(f"Model already loaded from :green[`{st.session_state.model_path}`]", icon="✅")
    else:
        st.session_state.model_path = utils.get_most_recent_model_path()
        st.session_state.model = load_model(st.session_state.model_path)
        st.toast(f"Loaded model from :green[`{st.session_state.model_path}`]", icon="🔄")


@st.cache_resource(max_entries=1)
def gen_diagram():
    model = utils.LSTMWithAttention()
    x = torch.zeros(1, utils.sequence_length, utils.input_size)
    digraph = make_dot(model(x), params=dict(model.named_parameters()))
    return digraph


@st.cache_data(max_entries=10)
def get_predictions(dataset_name, model_path):
    # Explicit check to ensure that the dataset and model in st.session_state
    # corresponds to those associated with the dataset_name and model_paht in the func args.
    # This enables caching based on dataset_name and model_path rather than the dataset and model themselves,
    # ensuring that the dataset and model in the st.session_state are safe to use.
    assert dataset_name == st.session_state.dataset_name
    assert model_path == st.session_state.model_path

    all_ground_truth = []
    all_predictions = []

    with torch.no_grad():
        for feature, label in st.session_state.dataloader:
            outputs = st.session_state.model(feature)

            # Use argmax to get class predictions if your outputs are probabilities
            predicted_classes = torch.argmax(outputs, dim=1)

            all_predictions.extend(predicted_classes.detach().numpy())
            all_ground_truth.extend(label.detach().numpy())

    df = pd.DataFrame({"ground_truth": all_ground_truth, "predicted": all_predictions})
    df["correct"] = df["ground_truth"] == df["predicted"]
    return df


@st.cache_data(max_entries=200)
def get_example_ground_truth_detail_data(idx, dataset_name):
    assert dataset_name == st.session_state.dataset_name

    feature, label = st.session_state.dataset[idx]

    feature_df = pd.DataFrame(feature)
    feature_dim_names = ["x", "y", "z"]
    feature_df.columns = feature_dim_names
    feature_df.index.name = "Timestep"
    feature_df = feature_df.reset_index()
    feature_df["Timestep"] = feature_df["Timestep"].apply(lambda i: f"{i:3d}")

    label = label.numpy().astype(int).item()

    return feature_df, label


@st.cache_data(max_entries=200)
def get_example_prediction_detail_data(idx, dataset_name, model_path):
    # Explicit check to ensure that the dataset and model in st.session_state
    # corresponds to those associated with the dataset_name and model_paht in the func args.
    # This enables caching based on dataset_name and model_path rather than the dataset and model themselves,
    # ensuring that the dataset and model in the st.session_state are safe to use.
    assert dataset_name == st.session_state.dataset_name
    assert model_path == st.session_state.model_path

    feature, label = st.session_state.dataset[idx]

    x = feature[None, :]
    output, attn = st.session_state.model.forward_with_attn(x)

    predicted_label = torch.argmax(output, dim=1).numpy().astype(int).item()
    predicted_proba = softmax(output, dim=1).detach().numpy()[0]

    predicted_attn = attn[0].detach().numpy()
    predicted_attn = np.mean(predicted_attn, axis=1)
    predicted_attn_df = pd.DataFrame(predicted_attn.T)
    predicted_attn_df.columns = head_cols

    return predicted_label, predicted_proba, predicted_attn_df


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


def first_time():
    load_dataset_callback(dataset_name=utils.DATASET_NAME)
    load_model_callback(model_path=utils.get_most_recent_model_path())
    st.session_state.init = True

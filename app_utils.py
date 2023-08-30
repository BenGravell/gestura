import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchviz import make_dot
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_fscore_support

import streamlit as st

import utils



################################################################################
# Rendering utils
################################################################################

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


################################################################################
# Data & model utils
################################################################################

@st.cache_data(max_entries=10)
def load_dataset(dataset_name="UWaveGestureLibrary"):
    return utils.load_dataset(dataset_name)


@st.cache_data(max_entries=10)
def load_test_dataset_and_noshuffle_dataloader(dataset_name="UWaveGestureLibrary"):
    _, _, dataset, _ = load_dataset(dataset_name)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    return dataset, dataloader


@st.cache_resource(max_entries=10)
def load_model(model_path):
    model = utils.load_checkpoint(model_path)["model"]
    model.eval()
    return model


def load_most_recent_model_callback():
    most_recent_model_path = utils.get_most_recent_model_path()
    if most_recent_model_path == st.session_state.get("model_path"):
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
def get_predictions(model_path):
    # Crude check to ensure that the model in st.session_state.model corresponds to model_path
    # This is to enable caching based on model_path rather than the model itself
    assert model_path == st.session_state.model_path

    all_ground_truth = []
    all_predictions = []

    dataset, dataloader = load_test_dataset_and_noshuffle_dataloader()

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


@st.cache_data(max_entries=10)
def get_confusion_matrix(model_path):
    # Crude check to ensure that the model in st.session_state.model corresponds to model_path
    # This is to enable caching based on model_path rather than the model itself
    assert model_path == st.session_state.model_path

    confusion_matrix = np.zeros((utils.output_size, utils.output_size), int)
    df = get_predictions(model_path)
    for idx, row in df.iterrows():
        confusion_matrix[row.ground_truth, row.predicted] += 1
    return confusion_matrix


def first_time():
    load_most_recent_model_callback()
    st.session_state.init = True
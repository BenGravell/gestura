import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config
import app_utils
import utils


st.set_page_config(page_title="Gestura", page_icon="🤌", layout="wide")

if not st.session_state.get("init"):
    app_utils.first_time()

app_utils.show_load_most_recent_model_button_in_sidebar()

dataset, dataloader = app_utils.load_test_dataset_and_noshuffle_dataloader()

st.title("Example Inspector")

st.caption(
    "Predictions and details for individual examples in the test set. None of the examples shown here have been exposed"
    " to the model during training."
)
options = [idx for idx in range(len(dataset))]
options_formatted = [f"{idx:03d} (label={dataset[idx][1]})" for idx in range(len(dataset))]
idx = st.selectbox("Example Index", options=options, format_func=lambda x: options_formatted[x])
feature, label = dataset[idx]

feature_df = pd.DataFrame(feature)
feature_dim_names = ["x", "y", "z"]
feature_df.columns = feature_dim_names
feature_df.index.name = "Timestep"
feature_df = feature_df.reset_index()
feature_df["Timestep"] = feature_df["Timestep"].apply(lambda i: f"{i:3d}")

x = feature[None, :]
output, attn = st.session_state.model.forward_with_attn(x)

predicted_label = torch.argmax(output, dim=1).numpy().astype(int).item()
label = label.numpy().astype(int).item()

labels = np.arange(utils.NUM_CLASSES).astype(float)
predicted = output.detach().numpy()[0]
predicted_proba = softmax(output, dim=1).detach().numpy()[0]

cols = st.columns([2, 3])
with cols[0]:
    st.header("3D Acceleration Trajectory", divider="blue")
    st.info(
        "This is **not** the literal 3D trajectory of *positions* in an inertial frame, but rather the 3D trajectory of"
        " *accelerations* as recorded by the accelerometer. Therefore, do **not** expect an easy matching with the"
        " ground truth gesture depiction.",
        icon="📢",
    )

    trace = go.Scatter3d(
        x=feature_df["x"],
        y=feature_df["y"],
        z=feature_df["z"],
        text=feature_df["Timestep"],
        hoverinfo="text+x+y+z",
        marker=dict(
            size=4,
            color=config.STREAMLIT_CONFIG["theme"]["primaryColor"],
            opacity=0.7,
        ),
        line=dict(color="black", width=3),
    )

    data = [trace]

    layout = go.Layout(height=500, margin=dict(l=0, r=0, b=0, t=0))

    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig, use_container_width=True)

with cols[1]:
    st.header("Time Series Features & Attention", divider="blue")

    predicted_attn = attn[0].detach().numpy()
    predicted_attn = np.mean(predicted_attn, axis=1)
    predicted_attn_df = pd.DataFrame(predicted_attn.T)
    head_cols = [f"Attention Head {i}" for i in range(utils.heads)]
    predicted_attn_df.columns = head_cols
    predicted_attn_df.index = feature_df.index

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Features", "Attention"),
        row_heights=[0.5, 0.5],
        vertical_spacing=0.1,
    )
    feat_cols = [f"Acceleration {feat}" for feat in ["x", "y", "z"]]
    line = px.line(
        feature_df.rename(columns={feat: f"Acceleration {feat}" for feat in ["x", "y", "z"]}),
        x="Timestep",
        y=feat_cols,
    )
    fig.add_traces(list(line.select_traces()), rows=1, cols=1)

    attn_line = px.line(feature_df.join(predicted_attn_df), x="Timestep", y=head_cols)
    fig.add_traces(list(attn_line.select_traces()), rows=2, cols=1)

    fig.update_layout(height=600, margin=dict(l=50, r=50, b=50, t=50))
    st.plotly_chart(fig, use_container_width=True)

st.header("Predicted Class Probabilties", divider="blue")
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
        source=app_utils.class_index_to_gesture_image_url(i),
        xref="x",
        yref="y",
        sizex=0.4,
        sizey=0.4,
        xanchor="center",
        yanchor="middle",
    )

st.plotly_chart(fig, use_container_width=True)

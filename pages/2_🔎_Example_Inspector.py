import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import constants
from train_config import TRAIN_CONFIG
import streamlit_config
import app_utils


def setup():
    st.set_page_config(page_title="Gestura", page_icon="🤌", layout="wide")

    if not st.session_state.get("init"):
        app_utils.first_time()

    app_utils.show_load_model_button_in_sidebar()


def main():
    setup()

    st.title("Example Inspector")

    st.caption(
        "Predictions and details for individual examples in the test set. None of the examples shown here have been"
        " exposed to the model during training."
    )

    labels_selected = st.multiselect(
        "Ground Truth Labels to Select From",
        options=constants.LABELS,
        default=constants.LABELS,
        format_func=lambda x: f"{constants.LABEL_TO_NAME_MAP[x]} ({x})",
    )

    if len(labels_selected) == 0:
        st.warning("No labels selected, select at least one to have examples to select from!")
        return

    options = [
        idx for idx in range(len(st.session_state.dataset)) if st.session_state.dataset[idx][1] in labels_selected
    ]

    def example_idx_to_label(idx):
        return int(st.session_state.dataset[idx][1])

    def example_label_to_gesture_name(label):
        return constants.LABEL_TO_NAME_MAP[label]

    def example_idx_to_selection_option(idx):
        label = example_idx_to_label(idx)
        gesture_name = example_label_to_gesture_name(label)
        return f"Index: {idx:03d}, Label: {label}, Gesture Name: {gesture_name}"

    idx = st.selectbox(
        "Select Example",
        options=options,
        format_func=example_idx_to_selection_option,
    )

    feature_df, label = app_utils.get_example_ground_truth_detail_data(idx, st.session_state.dataset_name)
    predicted_label, predicted_proba, predicted_attn_df = app_utils.get_example_prediction_detail_data(
        idx, st.session_state.dataset_name, st.session_state.model_path
    )

    cols = st.columns([2, 3])
    with cols[0]:
        st.header("3D Acceleration Trajectory", divider="blue")
        if st.toggle("Show Explanation of 3D Acceleration Trajectory", value=True):
            st.info(
                "This is **not** the literal 3D trajectory of *positions* in an inertial frame, but rather the 3D"
                " trajectory of *accelerations* as recorded by the accelerometer. Therefore, do **not** expect an easy"
                " matching with the ground truth gesture depiction based on visual comparison.",
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
                color=streamlit_config.STREAMLIT_CONFIG["theme"]["primaryColor"],
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
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("Features", "Attention"),
            row_heights=[0.5, 0.5],
            vertical_spacing=0.1,
        )

        # Line plot for features
        colors = app_utils.get_colors_from_colormap("plasma", n=3, start=0.2, end=0.8)
        color_sequence_map = {feat: colors[i] for i, feat in enumerate(["x", "y", "z"])}
        for feat in ["x", "y", "z"]:
            trace = go.Scatter(
                x=feature_df["Timestep"],
                y=feature_df[feat],
                mode="lines",
                name=f"Acceleration {feat.upper()}",
                line={"color": color_sequence_map[feat]},
                legendgroup="features",
                legendgrouptitle_text="Features",
            )
            fig.add_trace(trace, row=1, col=1)

        # Line plot for attention
        colors = app_utils.get_colors_from_colormap("Blues", n=TRAIN_CONFIG.num_heads, start=0.4, end=1.0)
        color_sequence_map = {head_col: colors[i] for i, head_col in enumerate(constants.HEAD_NAMES)}
        for head_col in constants.HEAD_NAMES:
            trace = go.Scatter(
                x=feature_df["Timestep"],
                y=predicted_attn_df[head_col],
                mode="lines",
                name=head_col,
                line={"color": color_sequence_map[head_col]},
                legendgroup="attention",
                legendgrouptitle_text="Attention",
            )
            fig.add_trace(trace, row=2, col=1)

        fig.update_layout(height=600, margin=dict(l=50, r=50, b=50, t=50))
        st.plotly_chart(fig, use_container_width=True)

    st.header("Predicted Class Probabilties", divider="blue")
    fig = px.bar(x=constants.LABELS, y=predicted_proba)

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
        font=dict(size=14, color=streamlit_config.STREAMLIT_CONFIG["theme"]["primaryColor"]),
        textangle=0,
        xanchor="center",
    )
    # Customize the layout
    fig.update_layout(
        xaxis_title="Class",
        yaxis_title="Value",
        xaxis=dict(
            tickvals=list(range(constants.NUM_CLASSES)), ticktext=[str(i) for i in range(constants.NUM_CLASSES)]
        ),
    )

    # Adjust the y-axis range to leave space for the images
    fig.update_yaxes(range=[-0.8, 1.0])

    for i in range(constants.NUM_CLASSES):
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


main()

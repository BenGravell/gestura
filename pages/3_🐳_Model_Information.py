import inspect

import streamlit as st

import app_utils
import utils

st.set_page_config(page_title="Gestura", page_icon="🤌", layout="wide")

if not st.session_state.get("init"):
    app_utils.first_time()


st.header("Model Architecture Summary", divider="blue")
st.write(
    "The model used for prediction is a Long Short-Term Memory (LSTM) neural network with multi-head scaled"
    " dot-product self-attention."
)

st.subheader("Long Short-Term Memory (LSTM)")
st.write(
    "LSTMs are a type of recurrent neural network (RNN) architecture. While vanilla RNNs can theoretically capture"
    " long-range dependencies in sequential data, they often struggle to do so in practice due to the vanishing and"
    " exploding gradient problems. LSTMs were designed to address these issues."
)
app_utils.expander_markdown_from_file("Key features of LSTMs", "help/lstm_key_features.md")
app_utils.expander_markdown_from_file("Handy References for LSTMs", "help/lstm_handy_references.md")

st.subheader("Self-Attention")
st.write(
    "Self-attention, especially as popularized by the Transformer architecture, is a mechanism that enables a"
    " neural network to focus on different parts of the input data relative to a particular position in the data,"
    " often in the context of sequences. "
)
app_utils.expander_markdown_from_file("Key features of Self-Attention", "help/self_attention_key_features.md")
app_utils.expander_markdown_from_file("Handy References for Self-Attention", "help/self_attention_handy_references.md")

st.subheader("Putting Recurrence & Self-Attention Together")
st.write(
    "The combination of self-attention and recurrent neural networks has been shown to be be more effective than"
    " either self-attention or recurrence individually in time-series classification tasks by [Katrompas,"
    ' Alexander, Theodoros Ntakouris, and Vangelis Metsis. *"Recurrence and self-attention vs the transformer for'
    ' time-series classification: a comparative study."* International Conference on Artificial Intelligence in'
    " Medicine. Cham: Springer International Publishing,"
    " 2022](https://link.springer.com/chapter/10.1007/978-3-031-09342-5_10)."
)

st.header("Model Architecture Details", divider="blue")

with st.expander("Model Architecture Diagram"):
    st.write("The full architecture of the model is shown by the graph below.")
    st.graphviz_chart(app_utils.gen_diagram(), use_container_width=True)

with st.expander("Model Architecture Code"):
    st.write("The PyTorch model class code is given below.")
    SelfAttention_code = inspect.getsource(utils.SelfAttention)
    LSTMWithAttention_code = inspect.getsource(utils.LSTMWithAttention)
    # model_architecture_code =
    st.code(SelfAttention_code)
    st.code(LSTMWithAttention_code)

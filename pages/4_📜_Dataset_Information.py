import streamlit as st

import app_utils
import utils


st.set_page_config(page_title="Gestura", page_icon="🤌", layout="wide")

if not st.session_state.get("init"):
    app_utils.first_time()

st.header("Data Description", anchor=False, divider="blue")

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
        st.image(app_utils.class_index_to_gesture_image_url(i, local=True), caption=f"Gesture {i}")
st.write(
    'Gesture vocabulary adopted from [C.S. Myers, L.R. Rabiner, *"A comparative study of several dynamic'
    ' time-warping algorithms for connected word recognition,"* The Bell System Technical Journal 60 (1981)'
    " 1389-1409](https://ieeexplore.ieee.org/document/6771178). The dot denotes the start, and the arrow denotes"
    " the end."
)

st.header("Source", anchor=False, divider="blue")
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

st.header("Download Link", anchor=False, divider="blue")
st.write("http://www.timeseriesclassification.com/aeon-toolkit/UWaveGestureLibrary.zip")

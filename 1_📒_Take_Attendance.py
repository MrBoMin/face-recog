import streamlit as st
from Home import face_rec
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import time

# Title for the app
st.subheader('Real-Time Attendance System 🖥️')

# Retrieve the data from Redis Database
with st.spinner('Retrieving Data from Redis DB ...🏃'):
    redis_face_db = face_rec.retrive_data(name='info:myanmar')

# Time settings
waitTime = 10  # time in seconds
setTime = time.time()
realtimepred = face_rec.RealTimePred()  # real-time prediction class

# Initialize session state for success message and camera state
if 'show_success' not in st.session_state:
    st.session_state['show_success'] = False
if 'camera_state' not in st.session_state:
    st.session_state['camera_state'] = True
if 'update_ui' not in st.session_state:
    st.session_state['update_ui'] = False

# streamlit-webrtc callback function
def video_frame_callback(frame):
    global setTime
    img = frame.to_ndarray(format="bgr24")  # 3-dimensional numpy array
    pred_img = realtimepred.face_prediction(
        img, 
        redis_face_db,
        'facial_features',
        ['Name', 'Role', 'Batch'],
        thresh=0.5
    )

    timenow = time.time()
    difftime = timenow - setTime
    if difftime >= waitTime:
        realtimepred.saveLogs_redis()
        setTime = time.time() 
        st.session_state['show_success'] = True
        st.session_state['camera_state'] = False 
        st.session_state['update_ui'] = True  

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="realtimePrediction",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    async_processing=True
)

# Function to handle UI updates and stopping the camera
def handle_ui_updates():
    if st.session_state['update_ui']:
        if not st.session_state['camera_state'] and webrtc_ctx.state.playing:
            webrtc_ctx.stop()
        if st.session_state['show_success']:
            st.success('Attendance successfully stored!')
            # Reset the flag after displaying success message
            st.session_state['show_success'] = False  
        st.session_state['update_ui'] = False  

# Call the function to handle UI updates
handle_ui_updates()

# Add margin around the DataFrame
st.markdown('<style> .dataframe { margin: 20px; } </style>', unsafe_allow_html=True)
st.dataframe(redis_face_db)

import streamlit as st
from Home import face_rec
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import time

st.subheader('Real-Time Attendance System 🖥️')

with st.spinner('Retrieving Data from Redis DB ...🏃'):
    redis_face_db = face_rec.retrive_data(name='info:myanmar')

waitTime = 10
setTime = time.time()
realtimepred = face_rec.RealTimePred()

if 'show_success' not in st.session_state:
    st.session_state['show_success'] = False
if 'camera_state' not in st.session_state:
    st.session_state['camera_state'] = True
if 'update_ui' not in st.session_state:
    st.session_state['update_ui'] = False

def video_frame_callback(frame):
    global setTime
    img = frame.to_ndarray(format="bgr24")
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

def handle_ui_updates():
    if st.session_state['update_ui']:
        if not st.session_state['camera_state'] and webrtc_ctx.state.playing:
            webrtc_ctx.stop()
        if st.session_state['show_success']:
            st.success('Attendance successfully stored!')
            st.session_state['show_success'] = False
        st.session_state['update_ui'] = False

handle_ui_updates()

st.markdown('<style> .dataframe { margin: 20px; } </style>', unsafe_allow_html=True)
st.dataframe(redis_face_db)

def run_app():
    st.subheader('Take Attendance')
    st.write("Content for Take Attendance App")

if __name__ == "__main__":
    run_app()

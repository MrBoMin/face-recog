import streamlit as st
from Home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import numpy as np

st.set_page_config(page_title='Registration Form')
st.subheader('Registration Form')

registration_form = face_rec.RegistrationForm()
sample_count = 0

def get_department():
    department = st.selectbox("Select your Department", options=('Software', 'Network', 'Mathematics', 'English'))
    return department

person_name = st.text_input(label='Name', placeholder='First & Last Name')
role = st.selectbox(label='Select your Role', options=('Student', 'Teacher'))
if role == 'Student':
    batch = st.number_input("Enter Batch Number:", min_value=0, max_value=100, value=78)
    st.write(f"You have selected the batch number: {batch}")
elif role == 'Teacher':
    batch = get_department()
    st.write(f"You have selected the department: {batch}")

def video_callback_func(frame):
    global sample_count, stop_stream
    img = frame.to_ndarray(format='bgr24')
    reg_img, embedding = registration_form.get_embedding(img)

    if embedding is not None:
        with open('face_embedding.txt', mode='ab') as f:
            np.savetxt(f, embedding)
        sample_count += 1

    return av.VideoFrame.from_ndarray(reg_img, format="bgr24")

webrtc_streamer(key="registration", video_frame_callback=video_callback_func)

if st.button('Submit'):
    return_val = registration_form.save_data_in_redis_db(person_name,role,batch)
    if return_val is True:
        st.success(f"{person_name} registered successfully")
    elif return_val == 'name_false':
        st.error('Please enter the name: Name cannot be empty or spaces')
    elif return_val == 'file_false':
        st.error('face_embedding.txt is not found. Please refresh the page and execute again.')

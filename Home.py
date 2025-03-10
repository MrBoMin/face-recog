import streamlit as st

st.set_page_config(page_title='Attendance System', layout='centered')

st.header('Attendance System using Face Recognition')

with st.spinner("Loading Models and Connecting to Redis db ...🏃"):
    import face_rec
    st.image("pho.png", caption="This is a sample image", use_column_width=True)

st.success('Model loaded successfully ✅')
st.success('Redis db successfully connected ✅')

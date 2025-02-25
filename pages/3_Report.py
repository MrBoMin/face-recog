import streamlit as st
from Home import face_rec
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go

st.set_page_config(page_title='Reporting',layout='wide')
st.subheader('Reporting')

name = 'attendance:logs'
def load_logs(name,end=-1):
    logs_list = face_rec.r.lrange(name,start=0,end=end)
    return logs_list

tab1, tab2, tab3,tab4 = st.tabs(['Attendance Report','Dashboard','Logs','Registered Data'])

with tab4:
    if st.button('Refresh Data'):
        with st.spinner('Retriving Data from Redis DB ...'):
            redis_face_db = face_rec.retrive_data(name='info:myanmar')
            st.dataframe(redis_face_db[['Name','Role','Batch']])

with tab3:
    if st.button('Refresh Logs'):
        st.write(load_logs(name=name))

with tab1:
    st.subheader('Attendance Report')
    log_lists = load_logs(name=name)
    convert_byte_to_string = lambda x: x.decode('utf-8')
    log_lists_string = list(map(convert_byte_to_string,log_lists))
    split_string = lambda x: x.split('@')
    logs_nested_list = list(map(split_string, log_lists_string))
    logs_df = pd.DataFrame(logs_nested_list, columns=['Name','Batch','Role','Timestamp'])
    print(logs_df)
    logs_df['Timestamp'] = pd.to_datetime(logs_df['Timestamp'])
    logs_df['Date'] = logs_df['Timestamp'].dt.date
    report_df = logs_df.groupby(by=['Date','Name','Role','Batch']).agg(
        In_time = pd.NamedAgg('Timestamp','min'),
        Out_time = pd.NamedAgg('Timestamp', 'max')
    ).reset_index()
    report_df['In_time'] = pd.to_datetime(report_df['In_time'])
    report_df['Out_time'] = pd.to_datetime(report_df['Out_time'])
    report_df['Duration'] = report_df['Out_time'] - report_df['In_time']
    all_dates = report_df['Date'].unique()
    name_role = report_df[['Name','Role','Batch']].drop_duplicates().values.tolist()
    print(name_role)
    date_name_role_zip = []
    for dt in all_dates:
        for name,role,batch in name_role:
            date_name_role_zip.append([dt,name,role,batch])
    print(date_name_role_zip)
    date_name_role_zip_df = pd.DataFrame(date_name_role_zip, columns=['Date','Name','Role','Batch'])
    date_name_role_zip_df = pd.merge(date_name_role_zip_df, report_df, how='left')
    date_name_role_zip_df['Duration_seconds'] = date_name_role_zip_df['Duration'].dt.seconds
    date_name_role_zip_df['Duration_hours'] = date_name_role_zip_df['Duration_seconds'] / (60 * 60)
    def status_maker(x):
        if pd.Series(x).isnull().all():
            return 'Absent'
        elif x>=0 and x< 1:
            return 'Absent (Less than 1 hour)'
        elif x >= 1:
            return 'Present'
    date_name_role_zip_df['Status'] = date_name_role_zip_df['Duration_hours'].apply(status_maker)
    st.write(date_name_role_zip_df)

with tab2:
    st.subheader("Dashboard")
    df = date_name_role_zip_df.copy()
    col1, col2 = st.columns(2)
    with col1:
        redis_face_db = face_rec.retrive_data(name='info:myanmar')
        total_students = redis_face_db.groupby(['Name', 'Batch']).size().reset_index()['Name'].nunique()
        fig = go.Figure(go.Indicator(
        mode="number",
        value=total_students,
        title="Total Students ",
        number={'font':{'size':120}}
        ))
        st.plotly_chart(fig)
    with col2:
        status_counts = date_name_role_zip_df['Status'].value_counts()
        fig_status = px.pie(values=status_counts, names=status_counts.index, title='Attendance Status')
        st.plotly_chart(fig_status)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    absent_count_by_batch = df.groupby(['Batch', 'Month'])['Status'].apply(lambda x: (x == 'Absent').sum()).reset_index()
    fig = px.line(absent_count_by_batch, x='Month', y='Status', color='Batch',
                title='Absent Count by Month',
                labels={'Month': 'Month', 'Status': 'Absent Count', 'Batch': 'Batch'})
    st.plotly_chart(fig)
    absent_count_by_batch = df.groupby('Batch')['Status'].apply(lambda x: (x == 'Absent').sum()).reset_index()
    fig = px.bar(absent_count_by_batch, x='Batch', y='Status', title='Count of Absent by Batch')
    fig.update_layout(xaxis_title='Batch', yaxis_title='Count of Absent')
    st.plotly_chart(fig)

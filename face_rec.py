import numpy as np
import pandas as pd
import cv2
import streamlit as st

import redis

# insight face
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
# time
import time
from datetime import datetime

import os




# Connect to Redis Client
hostname = 'redis-17339.c89.us-east-1-3.ec2.redns.redis-cloud.com'
portnumber = 17339
password = 'bZchykStnPj7x1RIQzpVpKjQWqAsvV5H'

r = redis.StrictRedis(host=hostname,
                      port=portnumber,
                      password=password)

# Retrive Data from database
def retrive_data(name):
    retrive_dict= r.hgetall(name)
    retrive_series = pd.Series(retrive_dict)
    retrive_series = retrive_series.apply(lambda x: np.frombuffer(x,dtype=np.float32))
    index = retrive_series.index
    index = list(map(lambda x: x.decode(), index))
    retrive_series.index = index
    retrive_df =  retrive_series.to_frame().reset_index()
    retrive_df.columns = ['name_role_batch','facial_features']
    retrive_df[['Name','Role','Batch']] = retrive_df['name_role_batch'].apply(lambda x: x.split('@')).apply(pd.Series)
    return retrive_df[['Name','Role','Batch','facial_features']]


# configure face analysis
faceapp = FaceAnalysis(name='buffalo_sc',root='insightface_model', providers = ['CPUExecutionProvider'])
faceapp.prepare(ctx_id = 0, det_size=(640,640), det_thresh = 0.5)

# ML Search Algorithm
def ml_search_algorithm(dataframe, feature_column, test_vector,
                        name_role=('Name', 'Role', 'Batch'), thresh=0.5):
    """
    Cosine similarity based search algorithm
    """
    # Step 1: Take the dataframe (collection of data)
    dataframe = dataframe.copy()

    # Step 2: Index face embedding from the dataframe and convert into array
    X_list = dataframe[feature_column].tolist()
    x = np.asarray(X_list)

    # Step 3: Calculate cosine similarity
    similar = pairwise.cosine_similarity(x, test_vector.reshape(1, -1))
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr

    # Step 4: Filter the data
    data_filter = dataframe.query(f'cosine >= {thresh}')
    if len(data_filter) > 0:
        # Step 5: Get the person name and role and batch
        data_filter.reset_index(drop=True, inplace=True)
        argmax = data_filter['cosine'].argmax()
        person_name, person_role, person_batch = data_filter.loc[argmax][name_role]


    else:
        person_name = 'Unknown'
        person_role = 'Unknown'
        person_batch = 'Unknown'

    return person_name, person_role, person_batch



class RealTimePred:
    def __init__(self):
        self.logs = dict(name=[], role=[],batch = [], current_time=[])

    def reset_dict(self):
        self.logs = dict(name=[], role=[],batch = [],current_time=[])

    def saveLogs_redis(self):
        # Step 1: Create a logs dataframe
        dataframe = pd.DataFrame(self.logs)
        # Step 2: Drop the duplicate information (distinct name)
        dataframe.drop_duplicates('name', inplace=True)
        # Step 3: Push data to Redis database (list)
        # Encode the data
        encoded_data = []
        for idx, row in dataframe.iterrows():
            if row['name'] != 'Unknown':
                encoded_data.append(f"{row['name']}@{row['batch']}@{row['role']}@{row['current_time']}")

        if len(encoded_data) > 0:
            r.lpush('attendance:logs', *encoded_data)

        self.reset_dict()

    def face_prediction(self, test_image, dataframe, feature_column,
                        name_role=('Name', 'Role', 'Batch'), thresh=0.5):
        # Step 1: Find the time
        current_time = str(datetime.now())

        # Step 2: Take the test image and apply to insight face
        results = faceapp.get(test_image, max_num=99999)
        test_copy = test_image.copy()
        # Step 3: Use for loop and extract each embedding and pass to ml_search_algorithm

        for res in results:
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embeddings = res['embedding']
            person_name, person_role, person_batch = ml_search_algorithm(dataframe,
                                                           feature_column,
                                                           test_vector=embeddings,
                                                           name_role=name_role,
                                                           thresh=thresh)
            if person_name == 'Unknown':
                color = (0, 0, 255)  # BGR
            else:
                color = (0, 255, 0)

            cv2.rectangle(test_copy, (x1, y1), (x2, y2), color)

            text_gen = person_name
            cv2.putText(test_copy, text_gen, (x1, y1), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            cv2.putText(test_copy, current_time, (x1, y2 + 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            # Save info in logs dict
            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['batch'].append(person_batch)
            self.logs['current_time'].append(current_time)

        return test_copy


#### Registration Form
class RegistrationForm:
    def __init__(self):
        self.sample = 0
    def reset(self):
        self.sample = 0

    def get_embedding(self,frame):
        # get results from insightface model
        results = faceapp.get(frame,max_num=1)
        embeddings = None
        for res in results:
            self.sample += 1
            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),1)
            # put text samples info
            text = f"samples = {self.sample}"
            cv2.putText(frame,text,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,0),2)

            # facial features
            embeddings = res['embedding']

        return frame, embeddings

    def save_data_in_redis_db(self,name,role,batch):
        # validation name
        if name is not None:
            if name.strip() != '':
                key = f'{name}@{role}@{batch}'
            else:
                return 'name_false'
        else:
            return 'name_false'

        # if face_embedding.txt exists
        if 'face_embedding.txt' not in os.listdir():
            return 'file_false'


        # step-1: load "face_embedding.txt"
        x_array = np.loadtxt('face_embedding.txt',dtype=np.float32) # flatten array

        # step-2: convert into array (proper shape)
        received_samples = int(x_array.size/512)
        x_array = x_array.reshape(received_samples,512)
        x_array = np.asarray(x_array)

        # step-3: cal. mean embeddings
        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()

        # step-4: save this into redis database
        # redis hashes
        r.hset(name='info:myanmar',key=key,value=x_mean_bytes)

        #
        os.remove('face_embedding.txt')
        self.reset()

        return True
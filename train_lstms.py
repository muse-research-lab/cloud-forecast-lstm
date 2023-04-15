import os
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

job_id = '113812204462'
job_label = '113'
columns_to_use = ['avg_cpu_usage', 'time']
time_col = 'time'
col = 'avg_cpu_usage'
time_unit = 'us'
freq = '5min'

# folder path
dir_path = r'input-large/' + job_id + '/'

# list to store files
list_of_files = []

# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        list_of_files.append(path)
           
#print('The folder ' + str(job_id) + ' contains the following files: ' + str(list_of_files))
    
csv_data_list = []
for i in list_of_files:
    type(i)
    dir = 'input-large/' +job_id + '/' + i 
    csv_data_list.append(pd.read_csv(filepath_or_buffer=dir, usecols=columns_to_use))

#print(job_id)
print(job_label)

data = []
for df in csv_data_list:
    df[time_col] = pd.to_datetime(
    df[time_col], unit=time_unit)
    df.set_index(time_col, inplace=True)
    df = df[~df.index.duplicated(keep="first")]
    df = df.resample(freq).mean()
    df = df.asfreq(freq=freq, method="ffill")
    data.append(df)

x_train = data[0][:880]['avg_cpu_usage']

a = min(x_train)
b = max(x_train)
x_train_normalised = []
for x in x_train:
    x_train_normalised.append(((x-a)/(b-a)))

n_features = 1
n_steps = 30
# split into samples
X, y = split_sequence(x_train_normalised, n_steps)
x_temp = X
X = X.reshape((X.shape[0], X.shape[1], n_features))
print(job_label)
model = Sequential()
model.add(LSTM(50, input_shape=(n_steps, n_features), activation='softmax'))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam', metrics=['mse'])
es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=20)
history = model.fit(x=X, y=y, validation_split = 0.3, epochs=1000, verbose=0, shuffle=False)
print(job_label)
model.save('job' + job_id + '/lstm_model/model')

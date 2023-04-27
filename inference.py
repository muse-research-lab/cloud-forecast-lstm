import os
import sys
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from tensorflow import keras

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

job_id = sys.argv[1]
job_label = sys.argv[1]
columns_to_use = ['avg_cpu_usage', 'time']
time_col = 'time'
col = 'avg_cpu_usage'
time_unit = 'us'
freq = '5min'

# folder path
dir_path = r'docs/' + job_id + '/'

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
    dir = 'docs/' +job_id + '/' + i 
    csv_data_list.append(pd.read_csv(filepath_or_buffer=dir, usecols=columns_to_use))

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

error = []
lstm_result_self = []

number_of_tasks = len(list_of_files)
avg_res = []
for i in range (0, number_of_tasks):
    x_test = data[i][:800]['avg_cpu_usage']
    n_features = 1
    n_steps = 30

    a = min(x_test)
    b = max(x_test)
    x_test_normalised = []
    for x in x_test:
        x_test_normalised.append(((x-a)/(b-a)))

    # split into samples
    X_test, y_test = split_sequence(x_test_normalised, n_steps)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
                
    model = keras.models.load_model('job' + job_id + '/lstm_model/model')
    m = model
    yh = m.predict(X_test, verbose=0)
                    
    fig, ax = plt.subplots(figsize=(8,4))
    fig.set_size_inches(70, 10.5)
    ax.set_ylabel('Value', size = 25)
    ax.set_xlabel('Time', size = 25)
    test_task = str(list_of_files[i]).split("-")[0]
    ax.set_title('Prediction of task ' + test_task + ' of job' + job_label +' using self model', size = 30)

    plt.xticks(size = 35)
    plt.yticks(size = 35)
                
    y_denorm = []
    for y in yh:
        y_denorm.append(y*(b-a) + a)
        
    result = mean_absolute_error(x_test_normalised[30:530], yh[0:500])   
    avg_res.append(result)
        
    bins = np.linspace(0, 500, 500)
    plt.plot(bins, y_denorm[0:500], alpha=0.6, color='blue',label='prediction')
    plt.plot(bins, x_test[30:530], alpha=0.6, color='darkorange',label='actual')
    plt.legend(loc="upper left", prop={'size': 30})
    plt.savefig('task ' + test_task + ' of job ' + job_label +' using self model' + '.png', bbox_inches='tight')
    plt.close()
    lstm_result_self.append(np.mean(np.array(avg_res)))
    error.append(np.std(avg_res))
            

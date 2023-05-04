# Cloud Resource Forecasting Using LSTM Neural Networks

This repository contains the code to deploy and train a LSTM model and do inference on test data, using timeseries of CPU usage from Google's trace dataset (2019).

## Repository Structure <br />

* `docs/` : contains the timeseries data that are used in the experiments. The data can be found in the respective folders, with the label of each job.
*  `train_lstms.py` : deploys and trains a LSTM. Takes one command line argument as input, which is the label of the job that the LSTM will be trained on. It comprises of 3 digits, e.g., '113'. The output created is a folder containing the pre-trained model. To run the script: `python train_lstms.py <job_id>`
* `inference.py` : infers a timeseries using a pre-trained LSTM model. Takes one command line argument as input, which is the label of the model that will be fetched to do the inference. It comprises of 3 digits, e.g., '113'. The output created is the infered timeseries and a graph of the predicted and the actual values. To run the script: `python inference.py <job_id>`

## Model Deployment & Training <br />

<img src="docs/images/lstm_deployment.png" width="600"/>
<img src="docs/images/data_structure.png" width="500"/>

We train one model for each job. We configure the same hyperparameters across all models for simplicity. 

## Model Inference <br />

For the inference, we choose one model to predict the rest of the tasks of the same job it was trained on. The results are plotted in the same graph as the actual timeseries, as shown below: 

<img src="docs/images/task_15_job_917_using_self_model.png" width="1000"/>
<img src="docs/images/task_82_job_113_using_self_model.png" width="1000"/>


## Paper Reference <br />
Georgia Christofidi, Konstantinos Papaioannou, and Thaleia Dimitra Doudali. 2023. <br /><br /> Toward Pattern-based Model Selection for Cloud Resource Forecasting. <br /><br /> In 3rd Workshop on Machine Learning and Systems (EuroMLSys ’23), May 8, 2023, Rome, Italy. <br /> <br /> 

## Licence <br />
The MIT License (MIT)
 
## Acknowledgements <br />
This work is part of the grant FJC2021-047102-I, funded by MCIN/AEI/10.13039/501100011033 and the European Union «NextGenerationEU»/PRTR.<br /> 
<img src="docs/images/micin-financiadoUEnextgeneration-prtr-aei-1.png" width="1000"/>

# Cloud resources forecasting using LSTM Neural Networks

This repository contains the code to deploy and train a LSTM model and do inference on test data, using timeseries of CPU usage from Google's trace dataset (2019).

## Repository Structure <br />

* `input/` : contains the timeseries data that are used in the experiments.
*  `train_lstms.py` : deploys and trains a LSTM. Takes one command line argument as input, which is the label of the job that the LSTM will be trained on. It comprises of 3 digits, e.g., '113'. The output created is a folder containing the pre-trained model.
* `inference.py` : infers a timeseries using a pre-trained LSTM model. Takes one command line argument as input, which is the label of the model that will be fetched to do the inference. It comprises of 3 digits, e.g., '113'. The output created is the infered timeseries and a graph of the predicted and the actual values.

## Model Deployment & Training <br />

![Screenshot](docs/images/lstm_deployment.png)
![Screenshot](docs/images/data_structure.png)

<img src="docs/images/lstm_deployment.png" width="500"/>
<img src="docs/images/data_structure.png" width="250"/>

## Model Inference <br />




## Paper Reference <br />
Georgia Christofidi, Konstantinos Papaioannou, and Thaleia Dimitra Doudali. 2023. <br /> Toward Pattern-based Model Selection for Cloud Resource Forecasting. <br /> In 3rd Workshop on Machine Learning and Systems (EuroMLSys ’23), May 8, 2023, Rome, Italy. ACM, New York, NY, USA, 8 pages. <br /> https://doi.org/10.1145/3578356.3592588 <br />
 
 ## License<br />

Copyright (c) 2023 Muse Lab. All rights reserved. <br />

Licensed under the MIT License.

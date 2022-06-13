# MATP-with-HEAT
This repo contains the code for our paper entitled "Multi-Agent Trajectory Prediction with Heterogeneous Edge-Enhanced Graph Attention Network", IEEE T-ITS, 2022.

## Data preprocessing
Run `bash datapre_run.sh` to process all the scenarios provided by the INTERACTION dataset.

## Models
Base model -> Heat model -> HeatIR model.

## Traning
Run `python it_all_train.py -m Heat` to train the one-channel HEAT-based trajectory predictor.

## Validation

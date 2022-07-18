# MATP-with-HEAT
This repo contains the code for our paper entitled "Multi-Agent Trajectory Prediction with Heterogeneous Edge-Enhanced Graph Attention Network", IEEE T-ITS, 2022.

## Install dependencies via pip.
`pip install -r requirements.txt`

## Data preprocessing
The strucutre of the raw INTERACTION Dataset can be found in `INTERACTION Dataset Tree.txt`.

To obtain the sorted dataset, please refer to 
[INTERPRET_challenge_regular_generalizability_track](https://github.com/interaction-dataset/INTERPRET_challenge_regular_generalizability_track). 

Run `bash datapre_run.sh` to process all the scenarios provided by the INTERACTION dataset.

## Models
Base model -> Heat model -> HeatIR model.

## Traning
Run `python it_all_train.py -m Heat` to train the one-channel HEAT-based trajectory predictor.

## Validation

## Citation
If you have found this work to be useful, please consider citing our paper:
```
@article{mo2022multi,
  title={Multi-agent trajectory prediction with heterogeneous edge-enhanced graph attention network},
  author={Mo, Xiaoyu and Huang, Zhiyu and Xing, Yang and Lv, Chen},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2022},
  publisher={IEEE}
}
```

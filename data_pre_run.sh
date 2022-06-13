#!/usr/bin/env bash
python it_all_data_pre.py -d train -s DR_USA_Roundabout_FT &&
python it_all_data_pre.py -d val -s DR_USA_Roundabout_FT &&

python it_all_data_pre.py -d train -s DR_CHN_Roundabout_LN &&
python it_all_data_pre.py -d val -s DR_CHN_Roundabout_LN &&

python it_all_data_pre.py -d train -s DR_DEU_Roundabout_OF &&
python it_all_data_pre.py -d val -s DR_DEU_Roundabout_OF &&

python it_all_data_pre.py -d train -s DR_USA_Roundabout_EP &&
python it_all_data_pre.py -d val -s DR_USA_Roundabout_EP &&

python it_all_data_pre.py -d train -s DR_USA_Roundabout_SR &&
python it_all_data_pre.py -d val -s DR_USA_Roundabout_SR &&

python it_all_data_pre.py -d train -s DR_CHN_Merging_ZS &&
python it_all_data_pre.py -d val -s DR_CHN_Merging_ZS &&

python it_all_data_pre.py -d train -s DR_DEU_Merging_MT &&
python it_all_data_pre.py -d val -s DR_DEU_Merging_MT &&

python it_all_data_pre.py -d train -s DR_USA_Intersection_EP0 &&
python it_all_data_pre.py -d val -s DR_USA_Intersection_EP0 &&

python it_all_data_pre.py -d train -s DR_USA_Intersection_EP1 &&
python it_all_data_pre.py -d val -s DR_USA_Intersection_EP1 &&

python it_all_data_pre.py -d train -s DR_USA_Intersection_GL &&
python it_all_data_pre.py -d val -s DR_USA_Intersection_GL &&

python it_all_data_pre.py -d train -s DR_USA_Intersection_MA &&
python it_all_data_pre.py -d val -s DR_USA_Intersection_MA 


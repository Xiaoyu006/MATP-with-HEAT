from __future__ import print_function
from datetime import date
import os
import sys
import argparse
import random
import pickle
import pprint
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

from TDE_Loss import Interaction_TDE_Loss

from torch_geometric.data import DataLoader

from it_heat_g_model import IT_Heat_Net
from it_heat_gir_model import IT_HeatIR_Net

from it_all_dataset import IT_ALL_MTP_dataset

import time


def train_a_model(model_to_tr, num_ep=1):
    model_to_tr.train()
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    train_running_loss = 0.0
    for i, data in enumerate(trainDataloader):
        data.x = data.x[:,:,0:args['input_size']]
        if torch.sum(torch.isnan(data.x))>0:
            print('nan in x')

        optimizer.zero_grad()
        # forward + backward + optimize
        fut_pred = model_to_tr(data.to(args['device']))
        masked_gt = data.y[data.tar_mask] # tar_mask veh_tar_mask
        masked_pred = fut_pred[data.tar_mask]
        
        train_l = Interaction_TDE_Loss(masked_pred, masked_gt, err_type=args['train_loss_type'])
        if torch.sum(torch.isnan(train_l))>0:
            print('nan in train loss')
            print(data.veh_tar_mask)

        train_l.backward()
        a = torch.nn.utils.clip_grad_norm_(model_to_tr.parameters(), 10)
        optimizer.step()
        train_running_loss += train_l.item()
        if i % 1000 == 999:    # print every 1000 mini-batches
            print('ep {}, {} batches, {} - {}'.format( num_ep, i + 1, args['train_loss_type'], round(train_running_loss / 1000, 4)))
            train_running_loss = 0.0
        
    return round(train_running_loss / (i%1000), 4)

def val_a_model(model_to_val):
    model_to_val.eval()
    with torch.no_grad():
        print('Testing no grad')
        val_running_loss = 0.0
        for i, data in enumerate(valDataloader):
            data.x = data.x[:,-args['in_length']:,0:args['input_size']]

            fut_pred = model_to_val(data.to(args['device']))

            masked_gt = data.y[data.tar_mask]
            masked_pred = fut_pred[data.tar_mask]

            val_l = Interaction_TDE_Loss(masked_pred, masked_gt, err_type='ADE')            
            val_running_loss += val_l.item()

    print('validation loss ADE [ {} ]'.format(round(val_running_loss / (i+1), 4)))
    return round(val_running_loss / (i+1), 4)

def save_obj_pkl(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
   
    pp = pprint.PrettyPrinter(indent=1)
    def parse_args(cmd_args):
        """ Parse arguments from command line input
        """
        parser = argparse.ArgumentParser(description='Training parameters')
        parser.add_argument('-m', '--modeltype', type=str, default='R', help="the model structure")
        parser.add_argument('-g', '--gnn', type=str, default='GAT', help="the GNN to be used")
        parser.add_argument('-r', '--rnn', type=str, default='GRU', help="the RNN to be used")
        parser.add_argument('-d', '--inputsize', type=int, default=4, help="the Number of data to be used")
        parser.add_argument('-b', '--histlength', type=int, default=10, help="length of history 10, 30, 50")
        parser.add_argument('-f', '--futlength', type=int, default=30, help="length of future 50")
        parser.add_argument('-k', '--gpu', type=str, default='0', help="the GPU to be used")
        parser.add_argument('-i', '--number', type=int, default=0, help="run times of the py script")

        parser.set_defaults(render=False)
        return parser.parse_args(cmd_args)

    # Parse arguments
    cmd_args = sys.argv[1:]
    cmd_args = parse_args(cmd_args)

    ## Network Arguments
    args = {}
    args['run_i'] = cmd_args.number
    args['random_seed'] = 1
    args['encoder_size'] = 64 
    args['decoder_size'] = 256 
    args['dyn_embedding_size'] = 64 
    args['train_epoches'] = 12

    args['heat_in_channels_node'] = 64
    args['heat_in_channels_edge_attr'] = 5
    args['heat_in_channels_edge_type'] = 6
    args['heat_edge_attr_emb_size'] = 64
    args['heat_edge_type_emb_size'] = 64
    args['heat_node_emb_size'] = 64
    args['heat_out_channels'] = 128
    args['heat_heads'] = 3
    args['heat_concat']=True
    
    args['in_length'] = cmd_args.histlength
    args['out_length'] = 30
    args['input_size'] = cmd_args.inputsize
    args['input_embedding_size'] = 32 
    args['date'] = date.today().strftime("%b-%d-%Y")
    args['data_size'] = 'all' 
    args['batch_size'] = 20 
    args['net_type'] = cmd_args.modeltype 
    args['train_loss_type'] = 'ADE' 
    args['enc_rnn_type'] = cmd_args.rnn 
    args['gnn_type'] = cmd_args.gnn 
    
    device = torch.device('cuda:{}'.format(cmd_args.gpu) if torch.cuda.is_available() else "cpu")
    args['device'] = device

    # Initialize network
    if args['net_type'] == 'Heat':
        print('loading {} model'.format(args['net_type']))
        train_net = IT_Heat_Net(args) 
    elif args['net_type'] == 'HeatIR':
        print('loading {} model'.format(args['net_type']))
        train_net = IT_HeatIR_Net(args)    # IT_HeatR_Net
    else:
        print('\nselect a proper model type!\n')
    
    train_net.to(args['device'])

    pp.pprint(args)
    print('{}, {}: {}-{}, {}'.format(args['date'], args['net_type'], args['gnn_type'], args['enc_rnn_type'], args['device']))
    
    
    ## Initialize optimizer 
    optimizer = torch.optim.Adam(train_net.parameters(),lr=0.001) 
    scheduler = MultiStepLR(optimizer, milestones=[1, 2, 4, 6], gamma=0.5)

    full_train_set = IT_ALL_MTP_dataset(data_path='/home/xy/heatmtp_it_data/', 
                                        scenario_type='ALL', data_split='train')
    val_set = IT_ALL_MTP_dataset(data_path='/home/xy/heatmtp_it_data/', 
                                        scenario_type='ALL', data_split='val')
    
    torch.set_num_threads(4)
    trainDataloader = DataLoader(full_train_set, batch_size=args['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    valDataloader = DataLoader(val_set, batch_size=args['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

    tic = time.time()
    Val_LOSS = []
    Train_LOSS = []
    min_val_loss = 10.0
    for ep in range(1, args['train_epoches']+1):
        train_loss_ep = train_a_model(train_net, num_ep=ep)
        val_loss_ep = val_a_model(train_net)

        Val_LOSS.append(val_loss_ep)
        Train_LOSS.append(train_loss_ep)
        scheduler.step()
        ## save model
        if val_loss_ep<min_val_loss:
            save_model_to_PATH = './it_all_models/{}_{}_{}_{}_h{}f{}_d{}_{}.tar'.format(args['date'], args['net_type'], args['gnn_type'], args['enc_rnn_type'], 
                                                                                 args['in_length'], args['out_length'], args['input_size'], args['run_i'])
            torch.save(train_net.state_dict(), save_model_to_PATH)
            min_val_loss = val_loss_ep

        with open('./it_all_models/{}-{}-{}-{}-h{}f{}-TRAINloss-d{}-{}.txt'.format(args['date'], args['net_type'], args['gnn_type'], args['enc_rnn_type'], 
                                                                            args['in_length'], args['out_length'], args['input_size'], args['run_i']), "w") as file:
            file.write(str(Train_LOSS))
        with open('./it_all_models/{}-{}-{}-{}-h{}f{}-VALloss-d{}-{}.txt'.format(args['date'], args['net_type'], args['gnn_type'], args['enc_rnn_type'], 
                                                                   args['in_length'], args['out_length'], args['input_size'], args['run_i']), "w") as file:
            file.write(str(Val_LOSS))
        save_obj_pkl(args, save_model_to_PATH.split('.tar')[0])


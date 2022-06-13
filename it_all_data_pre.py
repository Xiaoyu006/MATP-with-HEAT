""" 
Basic data preprocessor for INTERACTION dataset.
Each node has its own coordinate system.

data.x = [num_agent, agent_features]
data.y = [num_agent, agent_future_trajectories]
data.edge_index = [num_edges, 2] 
data.edge_attr = [num_edges, edge_attribute]
data.map = [map_width, map_length]
data.target_mask = [num_nodes, 1] 

edge_index will include:
    1. agent-agent edges
        1. relative position
        2. relative yaw angle
    2. map-agent edges
        1. agents position in the map
        2. agents yaw angle in the map

target_mask contains the boolean indicating whether the node is to be predicted. 
a node could be an agent or a map.
"""

import os
import sys
import random
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import argparse

import time
import math
# import torch_geometric
import scipy.io as scp
import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.ndimage.interpolation import rotate

import io
from torch_geometric.data import Data
from visualizations import map_vis_xy

class IT_DATA_PRE():
    def __init__(self, tracks=['.'], save_path='.',
                    seg_len=40, gap_len=1, hist_len=10, fut_len=30, map_pad_width=120):
        self.tracks = tracks
        self.save_path = save_path
        self.seg_len = seg_len
        self.gap_len = gap_len
        self.hist_len = hist_len
        self.fut_len = fut_len
        self.pad_pxl_width = map_pad_width
        self.pxl_lengths = {'x': 0.25, 'y': 0.25}
        self.node_type_to_indicator_vec = { 'car': torch.tensor([[0,0,1]]),
                                            'pedestrian/bicycle': torch.tensor([[0,1,0]]),
                                            'map': torch.tensor([[1,0,0]])}
    
    def set_track(self, track_num):
        ''' set the csv track file and read the track.csv file '''
        self.cur_track_name = self.tracks[track_num]
        print('\n')
        print(self.cur_track_name)
        # print(self.cur_track_name.split('_test.csv')[0])
        self.cur_map_path = './visualizations/osm_maps/{}.osm'.format(self.cur_track_name.split('.csv')[0].split('/')[-4])
        print('self.cur_map_path', self.cur_map_path)
        self.cur_map_png_path = './visualizations/map_png/{}_map.png'.format(self.cur_track_name.split('.csv')[0].split('/')[-4])
        print('self.cur_map_png_path', self.cur_map_png_path)
        self.map_img_np = self.read_img_to_numpy()

        print('map size {}'.format(self.map_img_np.shape))

        # self.padded_map_img_np = self.pad_map_img_in_numpy()
        self.map_limits_n_center = self.plot_map()
        # print(self.cur_track_name)
        self.cur_track = self.read_track()
        self.cur_agn_id_to_traj = self.get_agn_id_to_traj()

    def read_track(self):
        """ Read a track.csv file into a pandas dataframe. """
        track_df = pd.read_csv(self.cur_track_name)    
        print(track_df.head(5))
        ### Insert a new column call v_psi_rad
        vx_all = track_df['vx'].values
        vy_all = track_df['vy'].values

        vel_psi_rad = np.around(np.arctan2(vy_all, vx_all), decimals=3)
        
        track_df.insert(len(track_df.columns), 'vel_psi_rad', vel_psi_rad, allow_duplicates=False)
        # print(track_df.head(5))    
        return track_df # if case_id==None else track_df[track_df['case_id']==case_id]

    def get_agn_id_to_traj(self):
        ''' get the dict, where key is agent_id and value is the trajectory of this agent '''
        # print('getting agent to track dict')
        agent_id_to_traj = {}
        agn_IDs = set(self.cur_track['track_id'].values)
        for i, agn_id in enumerate(agn_IDs):
            agn_traj = self.cur_track[self.cur_track['track_id'] == agn_id][['frame_id', 'track_id', 'agent_type', 'x', 'y', 'vx', 'vy', 'psi_rad', 'vel_psi_rad']]
            agent_id_to_traj[agn_id] = agn_traj
        return agent_id_to_traj

    def Srotate(self, angle, valuex, valuey, pointx, pointy):
        sRotatex = (valuex-pointx)*math.cos(angle) + (valuey-pointy)*math.sin(angle) + pointx
        sRotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy
        return sRotatex, sRotatey

    def get_hist(self, agn_id, cur_frm_id):
        """ get the hist of agn_id from [cur_frm_id-9, cur_frm_id] rotated and translated to its own coordinate system.
            <agn_id, tar_agn_id, cur_frm_id> 
            return ( new_hist, raw_hist ) """
        start_frm_id = cur_frm_id - self.hist_len + 1
        traj = self.cur_agn_id_to_traj[agn_id]
        raw_hist = traj[(traj['frame_id']>=start_frm_id)
                        &(traj['frame_id']<=cur_frm_id)][['x', 'y', 'vx', 'vy']].values
        # print('raw hist {}'.format(raw_hist.shape))
        ## calculate psi_rad using vx and vy for both car and others.
        agn_cur_pos = traj[traj['frame_id']==cur_frm_id][['x', 'y']].values[0]
        agn_cur_psi_rad = traj[traj['frame_id']==cur_frm_id]['vel_psi_rad'].values

        ## translate, pad, and rotate the raw hist
        new_hist = raw_hist - np.insert(agn_cur_pos, 2, [0, 0])
        # print(f'new hist {new_hist.shape}')
        assert new_hist.shape[0]>0, f'new hist shape: {new_hist.shape}'
        if new_hist.shape[0] < self.hist_len:
            new_hist = np.pad(new_hist, pad_width=((self.hist_len-new_hist.shape[0], 0), (0,0)), mode='edge')
        if raw_hist.shape[0] < self.hist_len:
            raw_hist = np.pad(raw_hist, pad_width=((self.hist_len-raw_hist.shape[0], 0), (0,0)), mode='edge')
        new_hist[:,0], new_hist[:,1] = self.Srotate(agn_cur_psi_rad, new_hist[:,0], new_hist[:,1], 0, 0) # Rotated position w.r.t. the agent itself
        new_hist[:,2], new_hist[:,3] = self.Srotate(agn_cur_psi_rad, new_hist[:,2], new_hist[:,3], 0, 0) # Rotated velocity w.r.t. the agent itself

        # print('new hist: \n{} \nraw hsit: \n{}'.format(new_hist, raw_hist))
        return new_hist, raw_hist

    def get_fut(self, agn_id, cur_frm_id):
        """ get the hist of agn_id from [cur_frm_id, cur_frm_id+80] rotated and translated to its own coordinate system.
            <agn_id, tar_agn_id, cur_frm_id> 
            return ( new_fut, raw_fut, be_target )"""
        be_target = True
        fut_frm = cur_frm_id + self.fut_len
        traj = self.cur_agn_id_to_traj[agn_id]
        raw_fut = traj[(traj['frame_id']>cur_frm_id)
                       &(traj['frame_id']<=fut_frm)][['x', 'y']].values
        
        ## calculate psi_rad using vx and vy for both car and others.
        agn_cur_pos = traj[traj['frame_id']==cur_frm_id][['x', 'y']].values[0]
        agn_cur_psi_rad = traj[traj['frame_id']==cur_frm_id]['vel_psi_rad'].values

        ## translate, pad, and rotate the raw hist
        new_fut = raw_fut - agn_cur_pos

        # assert new_fut.shape[0]>0, f'new fut shape: {new_fut.shape}'
        if new_fut.shape[0] < self.fut_len:
            be_target = False
            new_fut = np.pad(new_fut, pad_width=((0, self.fut_len-new_fut.shape[0]), (0,0)), mode='empty')

        if raw_fut.shape[0] < self.fut_len:
            raw_fut = np.pad(raw_fut, pad_width=((0, self.fut_len-raw_fut.shape[0]), (0,0)), mode='empty')

        new_fut[:,0], new_fut[:,1] = self.Srotate(agn_cur_psi_rad, new_fut[:,0], new_fut[:,1], 0, 0) # Rotated position w.r.t. the agent itself

        # print('new_fut: \n{} \nraw fut: \n{}'.format(new_fut, raw_fut))
        return new_fut, raw_fut, be_target


    def get_nbrs(self, tar_agn_id, cur_frm_id, radii=30):
        """ get the nbrs of tar_agn within radii meters <tar_agn_id, cur_frm_id, radii=20> """
        
        Frame = self.cur_track[self.cur_track['frame_id']==cur_frm_id]
        
        
        ref_pos = Frame[Frame['track_id']==tar_agn_id][['x', 'y']].values[0]
        x_sqr = np.square(Frame[['x']].values - ref_pos[0])
        y_sqr = np.square(Frame[['y']].values - ref_pos[1])
        
        Frame.insert(len(Frame.columns), 'dist2tar', np.sqrt(x_sqr + y_sqr), False)
        return Frame[(Frame['dist2tar']>0.01) # exlude the target vehicle
                    &(Frame['dist2tar']<=radii)]['track_id'].values

    def plot_map(self):
        fig, axes = plt.subplots(1, 1)
        # plt.subplots_adjust(wspace=0, hspace=0)/
        fig.canvas.set_window_title("Interaction Dataset Visualization")
        lat_origin, lon_origin = 0. , 0.  # origin is necessary to correctly project the lat lon values in the osm file to the local
        map_vis_xy.draw_map_without_lanelet_xy(self.cur_map_path, axes, lat_origin, lon_origin)

        x_ax_limits, y_ax_limits = axes.get_xlim(), axes.get_ylim()
        plt.close()
        map_center_x = 0.5 * (x_ax_limits[1] + x_ax_limits[0])
        map_center_y = 0.5 * (y_ax_limits[1] + y_ax_limits[0])
        return {'map_xlim': x_ax_limits, 'map_ylim': y_ax_limits, 'map_center': [map_center_x, map_center_y]}

    def read_img_to_numpy(self):
        with Image.open(self.cur_map_png_path) as map_png_image:
            gray_image = ImageOps.grayscale(map_png_image)
        return np.asarray(gray_image)

    def graph_a_frame(self, frm_g, draw_g=False):
        DG = nx.DiGraph()
        """ construct the graph for a given frame [frm_g] """
        raw_v_IDs = frm_g['track_id'].values
        frm_id_g = frm_g['frame_id'].values[0]

        # Add Nodes, Pos, Lables and Edges
        Poses, Labels = {}, {}
        for v_id in raw_v_IDs:
            v_hist, _ = self.get_hist(agn_id=v_id, cur_frm_id=frm_id_g)
            DG.add_node(v_id, node_feature=v_hist)
            Poses[v_id] = frm_g[frm_g['track_id']== v_id][['x', 'y']].values[0]
            Labels[v_id] = str(v_id)
            v_nbrs = self.get_nbrs(tar_agn_id=v_id, cur_frm_id=frm_id_g, radii=30)
            for v_nbr_id in v_nbrs:
                DG.add_edge(v_nbr_id, v_id)
        if draw_g:
            raise NotImplementedError()
        return DG

    def get_data_a_frame(self, frm_g, cur_frm_id):
        """ given a frame, preprocess the frame to get a pyg data for this frame """
        
        ### Processing agents including car and pedestrian
        raw_v_IDs = frm_g['track_id'].values
        frm_id_g = frm_g['frame_id'].values[0]

        v_id_to_node_index = {}
        for i, v_id in enumerate(raw_v_IDs):
            v_id_to_node_index[v_id] = i

        ## Node features
        Nodes_f = torch.zeros((len(raw_v_IDs), self.hist_len, 4)) # save one place as map feature place holder.
        Fut_GT = torch.zeros((len(raw_v_IDs), self.fut_len, 2)) # save one place as map feature place holder.

        ## Edge
        Edges = torch.empty((2, 0)).long()
        Edges_attr = torch.empty((5, 0)) # [d_x, d_y, d_vx, d_vy, d_psi]
        Edges_type = torch.empty((6, 0)).long()

        ## Vehicle to map attributes
        Veh_Map_Attr = torch.empty((5, 0))
        
        ## Ground truth
        GT = torch.zeros((len(raw_v_IDs), self.fut_len, 2))

        ## Masks
        Tar_Mask = []
        Veh_Tar_Mask = [] # mask for target vehicles
        Veh_Mask = []
        Ped_Mask = []
        
        ## Map
        Map = torch.from_numpy(np.copy(self.map_img_np))
        Map_center = torch.tensor(self.map_limits_n_center['map_center'])

        ## Raw hist and fut
        Raw_hist = torch.zeros((len(raw_v_IDs), self.hist_len, 4)) # save one place as map feature place holder.
        Raw_fut = torch.zeros((len(raw_v_IDs), self.fut_len, 2)) # save one place as map feature place holder.

        for i, v_id in enumerate(raw_v_IDs):
            ## node feature
            v_hist, raw_h = self.get_hist(agn_id=v_id, cur_frm_id=frm_id_g)
            # print(v_hist.shape)
            Nodes_f[i] = torch.from_numpy(v_hist)
            Raw_hist[i] = torch.from_numpy(raw_h)

            ## v_id current state
            agn_cur_state = frm_g[frm_g['track_id']==v_id][['x', 'y', 'vx', 'vy', 'vel_psi_rad']].values[0]
            agn_cur_state = torch.from_numpy(agn_cur_state)
            agn_type = frm_g[frm_g['track_id']==v_id]['agent_type'].values[0]

            if agn_type == 'car':
                Veh_Mask.append(True)
                Ped_Mask.append(False)
            elif agn_type == 'pedestrian/bicycle':
                Veh_Mask.append(False)
                Ped_Mask.append(True)
            else:
                print(f'\n\n check the agent type {agn_type}\n\n')

            ## edge 
            v_nbrs = self.get_nbrs(tar_agn_id=v_id, cur_frm_id=frm_id_g, radii=30)
            v_node_index = v_id_to_node_index[v_id]
            # self loop
            self_edge = torch.tensor([[v_node_index], [v_node_index]])
            self_edge_attr = torch.tensor([0, 0, 0, 0, 0]).float().unsqueeze(dim=1)
            self_edge_type = torch.cat((self.node_type_to_indicator_vec[agn_type], self.node_type_to_indicator_vec[agn_type]), dim=1)
            Edges = torch.cat((Edges, self_edge), dim=1)
            Edges_attr = torch.cat((Edges_attr, self_edge_attr), dim=1)
            Edges_type = torch.cat((Edges_type, self_edge_type.transpose(0,1)), dim=1)

            for v_nbr_id in v_nbrs:
                nbr_v_node_index = v_id_to_node_index[v_nbr_id]
                
                # v_nbr_id current state
                nbr_cur_state = frm_g[frm_g['track_id']==v_nbr_id][['x', 'y', 'vx', 'vy', 'vel_psi_rad']].values[0]
                nbr_cur_state = torch.from_numpy(nbr_cur_state)
                nbr_type = frm_g[frm_g['track_id']==v_nbr_id]['agent_type'].values[0]
                
                ## edge
                edge = torch.tensor([[nbr_v_node_index], [v_node_index]])
                edge_attr = nbr_cur_state - agn_cur_state
                edge_attr = edge_attr.float().unsqueeze(dim=1)
                edge_type = torch.cat((self.node_type_to_indicator_vec[nbr_type], self.node_type_to_indicator_vec[agn_type]), dim=1)
                Edges = torch.cat((Edges, edge), dim=1)
                Edges_attr = torch.cat((Edges_attr, edge_attr), dim=1)
                Edges_type = torch.cat((Edges_type, edge_type.transpose(0,1)), dim=1)

            ## map
            veh_map_attr = agn_cur_state.float() - torch.cat((Map_center.float(), torch.tensor([0., 0., 0.])), dim=0)
            veh_map_attr = veh_map_attr.unsqueeze(dim=1)
            Veh_Map_Attr = torch.cat((Veh_Map_Attr, veh_map_attr), dim=1)

            # Edges = torch.cat((Edges, map_edge), dim=1)
            # Edges_attr = torch.cat((Edges_attr, map_edge_attr), dim=1)
            # Edges_type = torch.cat((Edges_type, map_edge_type.transpose(0,1)), dim=1)
            
            ## future trajecotries
            v_fut, raw_f, tar = self.get_fut(agn_id=v_id, cur_frm_id=frm_id_g)
            Fut_GT[i] = torch.from_numpy(v_fut)
            Raw_fut[i] = torch.from_numpy(raw_f)
            Tar_Mask.append(tar)
            if agn_type == 'car' and tar:
                Veh_Tar_Mask.append(True)
            else:
                Veh_Tar_Mask.append(False)
    
        Tar_Mask = torch.tensor(Tar_Mask)
        Veh_Tar_Mask = torch.tensor(Veh_Tar_Mask)
        Veh_Mask = torch.tensor(Veh_Mask)
        Ped_Mask = torch.tensor(Ped_Mask)
        
        pyg_data = Data(x=Nodes_f, y=Fut_GT, 
                        edge_index=Edges, edge_attr=Edges_attr, edge_type=Edges_type, veh_map_attr=Veh_Map_Attr,
                        tar_mask=Tar_Mask, veh_tar_mask=Veh_Tar_Mask, veh_mask=Veh_Mask, ped_mask=Ped_Mask,
                        raw_hists=Raw_hist, raw_futs=Raw_fut)

        if torch.sum(Veh_Tar_Mask)>1:
            pyg_data_name = '{}{}_f{}.pyg'.format(self.save_path, self.cur_track_name.split('.csv')[0].split('/')[-1], cur_frm_id)
            torch.save(pyg_data, pyg_data_name)
        return Nodes_f, Edges, Edges_type, Edges_attr, Fut_GT, Tar_Mask, Veh_Tar_Mask, Raw_hist, Raw_fut

    def process_a_frame(self, cur_frm_id):
        Frame = self.cur_track[self.cur_track['frame_id']==cur_frm_id]
        return self.get_data_a_frame(Frame, cur_frm_id)

    def parallel_pre_a_track_file(self, track_num):
        self.set_track(track_num=track_num)
        frame_numbers = list(set(self.cur_track['frame_id'].values))[10:-self.fut_len]
        # print(self.map_img_np.flags['WRITEABLE'])
        map_save_path = '/'.join(self.save_path.split('/')[:-2]) + '/MAP.pt'
        torch.save(torch.from_numpy(np.copy(self.map_img_np)), map_save_path)

        ##############################################################
        ## single-processing
        for i in frame_numbers:
            self.process_a_frame(i)
        #     break
        ##############################################################

        ##############################################################
        ## multi-processing
        # num_pros = 8 if True else 1
        # with Pool(processes=num_pros) as p:  # , maxtasksperchild=2
        #     p.map(self.process_a_frame, [i for i in frame_numbers])
        ##############################################################
    
    def preprocess_all(self):
        for i in range(len(self.tracks)):
            print('processing {}'.format(self.tracks[i]))
            self.parallel_pre_a_track_file(i)
            # break
        
def parse_args(cmd_args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='data preprocessing parameters')
    parser.add_argument('-d', '--predata', type=str, default='train', help="the data to preprocess")
    parser.add_argument('-s', '--scene', type=str, default='DR_USA_Roundabout_FT', help="the scenario")
    parser.set_defaults(render=False)
    return parser.parse_args(cmd_args)

def read_all_tracks(split_name='/train/', recorded_path='/home/xy/interaction_dataset/recorded_trackfiles/'):
    print('reading all tracks for {}...'.format(split_name))
    all_track_names = []
    all_recorded_track_names = os.listdir(recorded_path + split_name + 'sorted/')
    print(all_recorded_track_names)
    for track_name in all_recorded_track_names:
        track_path = recorded_path + split_name + 'sorted/'
        all_track_names.append(track_path + track_name)
    print('{} tracks for {}'.format(len(all_track_names), split_name))
    for i in range(len(all_track_names)):
        print(i, all_track_names[i])
    return all_track_names

if __name__ == '__main__':
    from multiprocessing import Pool
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    # Parse arguments
    cmd_args = sys.argv[1:]
    cmd_args = parse_args(cmd_args)

    ALL_TRACK_NAMES = read_all_tracks(split_name='/{}/'.format(cmd_args.predata), # train val
                                     recorded_path='/home/xy/interaction_dataset/recorded_trackfiles/{}'.format(cmd_args.scene)) # DR_USA_Roundabout_FT
    
    SAVE_TO = '/home/xy/heatmtp_it_data/{}/{}/'.format(cmd_args.scene, cmd_args.predata)
    if not os.path.exists(SAVE_TO):
        os.makedirs(SAVE_TO)

    DataPRE = IT_DATA_PRE(tracks = ALL_TRACK_NAMES, save_path=SAVE_TO)
    
    DataPRE.preprocess_all()

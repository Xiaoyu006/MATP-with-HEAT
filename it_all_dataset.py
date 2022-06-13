import os
import os.path as osp
import time
import torch
from torch_geometric.data import Dataset, DataLoader

# data
class IT_ALL_MTP_dataset(Dataset):
    def __init__(self, data_path='/home/xy/heatmtp_it_data/', scenario_type='Roundabout', data_split='train', hist_gap=1, fut_gap=1):
        'Initialization'
        super(IT_ALL_MTP_dataset).__init__()
        self.data_path = data_path
        self.scenario_type = scenario_type
        self.data_split = data_split
        self.hist_gap = hist_gap
        self.fut_gap = fut_gap

        self.read_in_datanames()
    def read_in_datanames(self):
        if self.scenario_type == 'ALL':
            self.scenario_names = [ p for p in os.listdir(self.data_path)]
        elif len(self.scenario_type.split('_'))==1:
            self.scenario_names = [ p for p in os.listdir(self.data_path) if p.split('_')[2]== self.scenario_type]
        else:
            self.scenario_names = [ self.scenario_type]

        self.data_names=[]
        for s in self.scenario_names:
            s_d_names = os.listdir('{}{}/{}'.format(self.data_path, s, self.data_split))
            s_d_names = ['{}{}/{}/{}'.format(self.data_path, s, self.data_split, d) for d in s_d_names]
            # print(len(s_d_names))
            self.data_names += s_d_names
        # print(len(self.data_names))

        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data_names)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.data_names[index]
        # print(ID)
        data_item = torch.load(osp.join(self.data_path, ID))
        # tic = time.time()
        data_item.edge_attr = data_item.edge_attr.transpose(0,1)
        data_item.edge_type = data_item.edge_type.transpose(0,1)
        data_item.veh_map_attr = data_item.veh_map_attr.transpose(0,1)
        ''' downsample x and y '''
        data_item.x = data_item.x[:, ::self.hist_gap, :]
        data_item.y = data_item.y[:, self.fut_gap-1::self.fut_gap, :]

        ''' remove raw hists and raw futs to save memory '''
        # data_item.raw_hists = torch.tensor([0])
        # data_item.raw_futs = torch.tensor([0])

        map_name = '{}{}/{}'.format(self.data_path, ID.split('/')[4], 'new_MAP.pt')
        # print(map_name)
        data_item.map = torch.load(map_name).unsqueeze(dim=0).unsqueeze(dim=0)
        # data_item.map = torch.load('/home/xy/heatmtp_it_data/{}/MAP.pt'.format(ID.split('/')[4]).format()).unsqueeze(dim=0).unsqueeze(dim=0)
        # print(data_item.map.shape)
        # print(time.time() - tic)
        return data_item

if __name__ == '__main__':
    import random
    dataset = IT_ALL_MTP_dataset(data_path='/home/xy/heatmtp_it_data/', scenario_type='ALL', data_split='val')
    print(dataset.__getitem__(random.randint(1, 40000)).veh_tar_mask)
    print('there are {} data in this dataset'.format(dataset.__len__()))
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    # # print(dataset.__len__())
    for d in loader:
        print(d.veh_tar_mask)
    #     print(f'veh mask: {d.veh_mask}')
    #     print(f'ped mask: {d.ped_mask}')
    #     print(f'vv_edge mask: {d.vv_edge_mask}')
    #     print(f'vp_edge mask: {d.vp_edge_mask}')
    #     print(f'pp_edge mask: {d.pp_edge_mask}')
    #     print(f'pv_edge mask: {d.pv_edge_mask}')
    #     print(f'tar_mask: {d.tar_mask}')
    #     print(f'veh_tar_mask: {d.veh_tar_mask}')

        break
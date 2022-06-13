import torch
from torch import nn

from it_heat_g_model import IT_Heat_Net
from heat import HEATlayer

class IT_HeatIR_Net(IT_Heat_Net):
    def __init__(self, args):
        super(IT_HeatIR_Net, self).__init__(args)
        self.args = args
        ## set CNN for map
        self.set_cnn()
        self.map_gate_fc = nn.Linear(self.args['heat_edge_attr_emb_size']+64, 64)
        self.map_gate_sgm = nn.Sigmoid()
        
        # Decoder LSTM
        self.veh_dec_rnn = torch.nn.LSTM(self.args['heat_out_channels']+self.args['encoder_size']+64, # 64 is the final size of map vector
                                         self.args['decoder_size'], 2, batch_first=True)
        self.ped_dec_rnn = torch.nn.LSTM(self.args['heat_out_channels']+self.args['encoder_size']+64,
                                         self.args['decoder_size'], 2, batch_first=True)
    
    def set_cnn(self):
        self.Map_CNN =nn.Sequential(nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(8, 12), stride=(4,6)),
                                    nn.LeakyReLU(0.1),
                                    nn.BatchNorm2d(8),
                                    nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(6, 6), stride=3),
                                    nn.LeakyReLU(0.1),
                                    nn.BatchNorm2d(16),
                                    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=2),
                                    nn.LeakyReLU(0.1),
                                    nn.BatchNorm2d(32),
                                    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 4), stride=2),
                                    nn.LeakyReLU(0.1),
                                    nn.BatchNorm2d(32)
                                    )

        self.Map_FC = nn.Sequential(nn.Linear(32*3*3, 128),
                                    nn.LeakyReLU(0.1),
                                    nn.Linear(128, 64),
                                    nn.LeakyReLU(0.1)
                                    )

    def Map_Encoder(self, map_np):
        map_batch = torch.sub(1, torch.div(map_np, 255)).float()
        map_1 = self.Map_CNN(map_batch)
        # print(map_1.shape)
        map_1 = self.Map_FC(map_1.view(-1, 32*3*3))
        return map_1
    
    def Map_Selector(self, map_vec, map_edge_attr):
        map_edge_cat = torch.cat((map_vec, map_edge_attr), dim=1)
        # print(map_edge_cat.shape)
        map_gate = self.map_gate_sgm(self.map_gate_fc(map_edge_cat))
        selected_map = map_vec.mul(map_gate)         
        return selected_map
    
    def forward(self, data_pyg):

        map_f_vec = self.Map_Encoder(data_pyg.map[:1])
        map_f_vec = map_f_vec.repeat(data_pyg.veh_map_attr.shape[0], 1, 1, 1)
        map_f_vec = map_f_vec.squeeze()

        fwd_Hist_Enc = self.RNN_Encoder(data_pyg.x, data_pyg.veh_mask, data_pyg.ped_mask)  # Encode
        fwd_tar_GAT_Enc = self.HEAT_Interaction( hist_enc=fwd_Hist_Enc,                     # Interaction
                                                edge_idx=data_pyg.edge_index, 
                                                edge_attr=data_pyg.edge_attr, 
                                                edge_type=data_pyg.edge_type.float(),
                                                veh_node_mask=data_pyg.veh_mask, 
                                                ped_node_mask=data_pyg.ped_mask)
        
        map_edge_attr = self.heat_conv1.edge_attr_emb(data_pyg.veh_map_attr)
        selected_map_f_vec = self.Map_Selector(map_f_vec, map_edge_attr)

        enc = torch.cat((fwd_Hist_Enc, fwd_tar_GAT_Enc, selected_map_f_vec), 1)   # Combine Individual, Interaction, and Map features
        fut_pred = self.decode(enc, data_pyg.veh_mask, data_pyg.ped_mask)                   # Decode
        return fut_pred
    
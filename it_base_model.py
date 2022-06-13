import torch
import numpy as np
class IT_Base_Net(torch.nn.Module):
    ''' 
        Shared layers:
            self.ip_emb
            self.enc_rn
            self.dyn_emb
            self.op
            self.leaky_relu

            self.RNN_Encoder
            self.decode
        '''
    def __init__(self, args):
        super(IT_Base_Net, self).__init__()
        self.args = args
        
        # Input embedding layer
        self.ip_emb = torch.nn.Linear(self.args['input_size'], self.args['input_embedding_size'])
        # Encoder LSTM
        self.veh_enc_rnn = torch.nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        self.ped_enc_rnn = torch.nn.GRU(self.args['input_embedding_size'], self.args['encoder_size'], 1, batch_first=True)
        # # Vehicle dynamics embedding
        self.dyn_emb = torch.nn.Linear(self.args['encoder_size'], self.args['dyn_embedding_size'])
        # Decoder LSTM
        self.veh_dec_rnn = torch.nn.LSTM(2*self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
        self.ped_dec_rnn = torch.nn.LSTM(2*self.args['encoder_size'], self.args['decoder_size'], 2, batch_first=True)
        # Output layers:
        self.op = torch.nn.Linear(self.args['decoder_size'], 2)
        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
    
    def RNN_Encoder(self, Hist, veh_mask, ped_mask):
        """ Encode sequential features of all considered vehicles 
            Hist: history trajectory of all vehicles
        """
        if torch.sum(veh_mask) == veh_mask.shape[0]:
            _, veh_Hist_Enc = self.veh_enc_rnn(self.leaky_relu(self.ip_emb(Hist)))
            veh_Hist_Enc = veh_Hist_Enc.view(veh_Hist_Enc.shape[1],veh_Hist_Enc.shape[2])
            Hist_Enc = self.leaky_relu(veh_Hist_Enc)
            return Hist_Enc
        elif torch.sum(ped_mask) == ped_mask.shape[0]:
            _, ped_Hist_Enc = self.ped_enc_rnn(self.leaky_relu(self.ip_emb(Hist[ped_mask])))
            ped_Hist_Enc = ped_Hist_Enc.view(ped_Hist_Enc.shape[1],ped_Hist_Enc.shape[2])
            Hist_Enc = self.leaky_relu(ped_Hist_Enc)
            return Hist_Enc
        else:
            Hist_Enc = torch.zeros((Hist.shape[0], self.args['encoder_size'])).to(ped_mask.device)
            _, veh_Hist_Enc = self.veh_enc_rnn(self.leaky_relu(self.ip_emb(Hist[veh_mask])))
            _, ped_Hist_Enc = self.ped_enc_rnn(self.leaky_relu(self.ip_emb(Hist[ped_mask])))
            Hist_Enc[veh_mask] = veh_Hist_Enc.view(veh_Hist_Enc.shape[1],veh_Hist_Enc.shape[2])
            Hist_Enc[ped_mask] = ped_Hist_Enc.view(ped_Hist_Enc.shape[1],ped_Hist_Enc.shape[2])
            Hist_Enc = self.leaky_relu(Hist_Enc)
            return Hist_Enc

    def forward(self, data_pyg):
        raiseNotImplementedError("forward is not implemented in IT_Base_Net!")
        
    def decode(self, enc, veh_mask, ped_mask):
        enc = enc.unsqueeze(1)
        enc = enc.repeat(1, self.args['out_length'], 1)
        ####################################
        if torch.sum(veh_mask) == veh_mask.shape[0]:
            Hist_Dec, _ = self.veh_dec_rnn(enc[veh_mask])
        elif torch.sum(ped_mask) == ped_mask.shape[0]:
            Hist_Dec, _ = self.ped_dec_rnn(enc[ped_mask])
        else:
            Hist_Dec = torch.zeros((enc.shape[0], self.args['out_length'], self.args['decoder_size'])).to(ped_mask.device)
            veh_dec, _ = self.veh_dec_rnn(enc[veh_mask])
            ped_dec, _ = self.ped_dec_rnn(enc[ped_mask])
            Hist_Dec[veh_mask] = veh_dec
            Hist_Dec[ped_mask] = ped_dec
        ####################################
        fut_pred = self.op(Hist_Dec)
        return fut_pred
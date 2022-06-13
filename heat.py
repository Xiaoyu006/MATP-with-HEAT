''' This is the first version of heat. it provides an edge emb for each kind of edge. the number of edge embedding grows as the number of types.
    It is better to set an edge attribute emb and an edge type emb then concatenate them. In this case, only two embeddings are needed.

'''
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F

class HEATlayer(nn.Module):
    ''' 
        1.  type specific transformation for nodes of different types: (vehicle, pedestrian/bicycle).
            transform nodes from different vector space to the same vector space.
            here we consider 2 types of nodes.
        2.  type specific transformation for edges of different types: (V->V, V->P, P->V, P->P).
            here we consider 4 types of edges.
        3.  node and edge feature combination (for V, P).
        4.  score the combinated or selected features and attention.
        5.  update the node feature.
            
    '''
    def __init__(self, in_channels_node=32, in_channels_edge_attr=2, in_channels_edge_type=2, edge_attr_emb_size=32, edge_type_emb_size=32, node_emb_size=32, out_channels=32, heads=3, concat=True):
        super(HEATlayer, self).__init__()
        ## Parameters
        self.in_channels_node = in_channels_node # 32
        self.in_channels_edge_attr = in_channels_edge_attr # 2
        self.in_channels_edge_type = in_channels_edge_type # 2
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.edge_attr_emb_size = edge_attr_emb_size
        self.edge_type_emb_size = edge_type_emb_size
        self.node_emb_size = node_emb_size

        #### Layers ####
        ## Embeddings 
        self.set_node_emb()
        self.set_edge_emb()

        # Transform the concatenated edge_nbrs feature to out_channels to update the next node feature
        self.node_update_emb = nn.Linear(self.edge_attr_emb_size + self.node_emb_size, self.out_channels, bias=False) 

        ## Attention
        self.attention_nn = nn.Linear(self.edge_attr_emb_size + self.edge_type_emb_size + 2*self.node_emb_size, 1*self.heads, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.soft_max = nn.Softmax(dim=1)
    
    def set_node_emb(self):
        ''' assume that different nodes have the same dimension, but different vector space. '''
        self.veh_node_feat_emb = nn.Linear(self.in_channels_node, self.node_emb_size, bias=False) # W_v
        self.ped_node_feat_emb = nn.Linear(self.in_channels_node, self.node_emb_size, bias=False) # W_p

    def set_edge_emb(self):
        ''' assume that different edges have the same dimension, but different vector space. '''
        self.edge_attr_emb = nn.Linear(self.in_channels_edge_attr, self.edge_attr_emb_size, bias=False) 
        self.edge_type_emb = nn.Linear(self.in_channels_edge_type, self.edge_type_emb_size, bias=False) 

    def embed_nodes(self, node_features, veh_node_mask, ped_node_mask):
        emb_node_features = torch.zeros(node_features.shape[0], self.node_emb_size).to(ped_node_mask.device)
        emb_node_features[veh_node_mask] = self.veh_node_feat_emb(node_features[veh_node_mask])
        emb_node_features[ped_node_mask] = self.ped_node_feat_emb(node_features[ped_node_mask])
        return emb_node_features
    
    def embed_edges(self, edge_attrs, edge_types):
        ''' embed edge attributes and edge types respectively and combine them as the edge feature. '''
        emb_edge_attributes = self.leaky_relu(self.edge_attr_emb(edge_attrs))
        emb_edge_types = self.leaky_relu(self.edge_type_emb(edge_types))
        return emb_edge_attributes, emb_edge_types
    
    def forward(self, node_f, edge_index, edge_attr, edge_type, veh_node_mask, ped_node_mask):
        """
        Args:
            node_f ([num_node, in_channels_nodeeature])
            edge_index ([2, number_edge])
            edge_attr ([number_edge, len_edge_feature])
        """
        emb_edge_attr, emb_edge_type = self.embed_edges(edge_attr, edge_type.float())

        emb_edge_f = torch.cat((emb_edge_attr, emb_edge_type), dim=1)

        size = torch.Size([node_f.shape[0], node_f.shape[0]] + [self.edge_attr_emb_size+self.edge_type_emb_size])
        emb_edge_f = torch.sparse_coo_tensor(edge_index, emb_edge_f, size).to_dense()
        
        # expand node feature to be with shape: [num_node, num_nbrs==num_node, node_feature_Wh]
        emb_node_f = self.leaky_relu(self.embed_nodes(node_f, veh_node_mask, ped_node_mask))
        nbrs_exp_emb_node_f = emb_node_f.unsqueeze(dim=0).repeat(node_f.shape[0], 1, 1)
        # concatenate nbrs features and edge features
        cat_edge_nbr_f_Weh = torch.cat((emb_edge_f, nbrs_exp_emb_node_f), dim=2)

        ##################################################
        # 
        # Update node feature with cat_node_edge_feature 
        # 
        ##################################################

        # 1. cat target node feature with edge (attr & type) with nbrs features and calculate scores
        tar_exp_emb_node_f = emb_node_f.unsqueeze(dim=1).repeat(1, node_f.shape[0], 1)
        cat_tar_edge_nbr_f_Wheh = torch.cat((tar_exp_emb_node_f, cat_edge_nbr_f_Weh), dim=2) # [num_nodes, num_nbrs, len_tar_f + in_channels_edge + len_nbr_f]
        
        scores = self.leaky_relu(self.attention_nn(cat_tar_edge_nbr_f_Wheh)) # [num_nodes, num_nbrs]
        
        # 2. select scores according to adjacent matrix, set scores where there is no edge to '-inf' or -10000
        adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]).to(ped_node_mask.device), torch.Size([node_f.shape[0], node_f.shape[0]])).to_dense().to(ped_node_mask.device)
        adj = adj.unsqueeze(dim=2).repeat(1,1,self.heads)
        
        scores_nbrs = torch.where(adj==0.0, torch.ones_like(scores) * -10000, scores)
        
        attention_nbrs = self.soft_max(scores_nbrs).unsqueeze(dim=3) # [num_nodes, num_nbrs, heads, 1]]

        attention_nbrs = attention_nbrs.repeat(1, 1, 1, self.out_channels)

        # 3. update node feature consider only edge attr and node feature, throw out edge type
        size = torch.Size([node_f.shape[0], node_f.shape[0]] + [self.edge_attr_emb_size])
        emb_edge_attr = torch.sparse_coo_tensor(edge_index, emb_edge_attr, size).to_dense()

        cat_edge_attr_nbr_feat = torch.cat((emb_edge_attr, nbrs_exp_emb_node_f), dim=2)
        cat_edge_attr_nbr_feat = cat_edge_attr_nbr_feat.unsqueeze(dim=2).repeat(1, 1, self.heads, 1)
        cat_edge_attr_nbr_feat = self.leaky_relu(self.node_update_emb(cat_edge_attr_nbr_feat))

        next_node_f = torch.sum(torch.mul(attention_nbrs,  cat_edge_attr_nbr_feat), dim=1)

        if self.concat:
            next_node_f = torch.flatten(next_node_f, start_dim=1)
        else:
            next_node_f = torch.mean(next_node_f, dim=1)
        
        return next_node_f

if __name__ == '__main__':
    heatlayer = HEATlayer(in_channels_node=31, in_channels_edge=2)
    
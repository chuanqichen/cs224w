import torch
from torch_geometric.utils import to_networkx
from ogb.linkproppred import PygLinkPropPredDataset
import networkx as nx
import json 
import pickle 

feat_dataset = PygLinkPropPredDataset(name='ogbl-ddi')
feat_data = feat_dataset[0]
print("Loading DDI done...")

G = to_networkx(feat_data, to_undirected=True)
print("To NetworkX done...")

node_centrality = nx.algorithms.centrality.betweenness_centrality(G)
print("Betweenness Centrality:", node_centrality)

feature_dict = dict()
feature_dict['betweenness_centrality'] = node_centrality

with open("ddi_node_centrality.pkl", "wb") as f:
    pickle.dump(feature_dict, f)


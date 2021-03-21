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

node_squares_clusters = nx.square_clustering(G)
print("Square Clustering:", node_squares_clusters)

feature_dict = dict()
feature_dict['square_clustering'] = node_squares_clusters

with open("ddi_SquareClustering.pkl", "wb") as f:
    pickle.dump(feature_dict, f)


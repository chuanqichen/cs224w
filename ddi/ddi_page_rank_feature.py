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

node_pagerank = nx.algorithms.link_analysis.pagerank_alg.pagerank(G)
print("Pagerank:", node_pagerank)

feature_dict = dict()
feature_dict['pagerank'] = node_pagerank

with open("ddi_page_rank_feature.pkl", "wb") as f:
    pickle.dump(feature_dict, f)


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

node_clusters = nx.clustering(G)    
print("Clustering:", node_clusters)

node_squares_clusters = nx.square_clustering(G)
print("Square Clustering:", node_squares_clusters)

node_deg = nx.generalized_degree(G)
print("Generalized Degree:", node_deg)

node_centrality = nx.algorithms.centrality.betweenness_centrality(G)
print("Betweenness Centrality:", node_centrality)

node_pagerank = nx.algorithms.link_analysis.pagerank_alg.pagerank(G)
print("Pagerank:", node_pagerank)

feature_dict = dict()
feature_dict['clustering'] = node_clusters
feature_dict['square_clustering'] = node_squares_clusters
feature_dict['generalized_degree'] = node_deg
feature_dict['betweenness_centrality'] = node_centrality
feature_dict['pagerank'] = node_pagerank

with open("features.pkl", "wb") as f:
    pickle.dump(feature_dict, f)


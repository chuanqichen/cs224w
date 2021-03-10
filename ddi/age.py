import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
from torch_geometric.utils.sparse import dense_to_sparse

from layers import *
from utils import *

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import os
from graph_global_attention_layer import LowRankAttention, weight_init
from logger import Logger

import scipy.sparse as sp
from sklearn.preprocessing import normalize

class LinTrans(nn.Module):
    def __init__(self, layers, dims):
        super(LinTrans, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
        self.dcs = SampleDecoder(act=lambda x: x)

    def scale(self, z):
        
        zmax = z.max(dim=1, keepdim=True)[0]
        zmin = z.min(dim=1, keepdim=True)[0]
        z_std = (z - zmin) / (zmax - zmin)
        z_scaled = z_std
    
        return z_scaled

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters() 

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        out = self.scale(out)
        out = F.normalize(out)
        return out

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)

def train(model,  inx, pos_inds_cuda, neg_inds, adj, split_edge,
                         optimizer, batch_size, cpu):

    #row, col, _ = adj_t.coo()
    #_, coo = dense_to_sparse(adj_t)
    n_nodes = adj.shape[0]

    coo = adj
    edge_index = torch.stack([coo[0], coo[1]], dim=0).to_dense()

    model.train()

    print(batch_size)
    print(pos_inds_cuda.shape)
    print(pos_inds_cuda.shape[0])
    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_inds_cuda.shape[0]), batch_size=batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        print("Perm size", perm.shape)

        if cpu:
            sampled_neg = torch.LongTensor(np.random.choice(neg_inds, size=batch_size))
        else:
            sampled_neg = torch.LongTensor(np.random.choice(neg_inds, size=batch_size)).cuda()

        sampled_inds = torch.cat((pos_inds_cuda[perm], sampled_neg), 0)
        
        x_idx = sampled_inds // n_nodes
        y_idx = sampled_inds % n_nodes

        x = torch.index_select(inx, 0, x_idx)
        y = torch.index_select(inx, 0, y_idx)
        print("x shape", x.shape)
        print("y shape", y.shape)

        zx = model(x)
        zy = model(y)

        print("Zx shape", zx.shape)
        print("Zy shape", zy.shape)
        
        if cpu:
            batch_label = torch.cat((torch.ones(batch_size), torch.zeros(batch_size)))
        else:
            batch_label = torch.cat((torch.ones(batch_size), torch.zeros(batch_size))).cuda()

        batch_pred = model.dcs(zx, zy)
        loss = F.binary_cross_entropy_with_logits(batch_pred, batch_label)
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(x, 1.0)
        # torch.nn.utils.clip_grad_norm_(y, 1.0)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, x, adj_t, split_edge, evaluator, batch_size):
    model.eval()

    h = model(x)

    pos_train_edge = split_edge['eval_train']['edge'].to(x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 20, 30]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results

def gpu_setup(gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  
    device = torch.device("cuda")
    return device

def update_similarity(z, upper_threshold, lower_treshold, pos_num, neg_num):
    f_adj = np.matmul(z, np.transpose(z))
    cosine = f_adj
    cosine = cosine.reshape([-1,])
    pos_num = round(upper_threshold * len(cosine))
    neg_num = round((1-lower_treshold) * len(cosine))
    
    pos_inds = np.argpartition(-cosine, pos_num)[:pos_num]
    neg_inds = np.argpartition(cosine, neg_num)[:neg_num]
    
    return np.array(pos_inds), np.array(neg_inds)

def update_threshold(upper_threshold, lower_treshold, up_eta, low_eta):
    upth = upper_threshold + up_eta
    lowth = lower_treshold + low_eta
    return upth, lowth

def main():
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    parser.add_argument('--cpu', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_gnn_layers', type=int, default=2)
    parser.add_argument('--num_linear_layers', type=int, default=1)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--age_dims', type=int, default=[500], help='Number of units in hidden layer 1.')
    parser.add_argument('--upth_st', type=float, default=0.0011, help='Upper Threshold start.')
    parser.add_argument('--lowth_st', type=float, default=0.1, help='Lower Threshold start.')
    parser.add_argument('--upth_ed', type=float, default=0.001, help='Upper Threshold end.')
    parser.add_argument('--lowth_ed', type=float, default=0.5, help='Lower Threshold end.')
    parser.add_argument('--upd', type=int, default=10, help='Update epoch.')

    args = parser.parse_args()
    print("Arguments", args)
    device = gpu_setup(args.gpu_id)

    dataset = PygLinkPropPredDataset(name='ogbl-ddi',
                                     transform=T.ToDense())
    data = dataset[0]
    print("[DDI INFO] DDI Graph is undirected")

    cpu = args.cpu
    if cpu:
        adj = data.adj
    else:
        adj = data.adj.to(device)
        
    split_edge = dataset.get_edge_split()

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['edge'].size(0))
    idx = idx[:split_edge['valid']['edge'].size(0)]
    split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}

    print("split edge shape", idx.shape)
    print("Adjacency Matrix shape", adj.shape)
    
    # [AGE] store original adj matrix (without diag entries) for later
    # TODO
    diag_ind = np.diag_indices(adj.shape[0])
    if cpu:
        adj[diag_ind[0], diag_ind[1]] = torch.zeros(adj.shape[0])
    else:
        adj[diag_ind[0], diag_ind[1]] = torch.zeros(adj.shape[0]).to(device)

    # Reconstruct adj_train from split_edge['train']
    train_edges = split_edge['train']['edge']
    print("train edge shape", train_edges.shape)
    print("[DDI INFO] only one set of edges in split_edge['train']['edge']")

    n_train_edges = train_edges.shape[0]
    adj_data = torch.ones(n_train_edges)
    adj_train = torch.sparse_coo_tensor(train_edges.transpose(1, 0), adj_data, (adj.shape)) 
    adj_train = adj_train + adj_train.transpose(1, 0)
    adj = adj_train
    n = adj.shape[0]
    
    adj_numpy = adj.to_dense().cpu().detach().numpy()

    adj_norm_s = preprocess_graph(adj_numpy, args.num_gnn_layers, norm='sym', renorm=True)

    # [AGE] Feature augmentation
    # FIXME: Change feature augmentation vector later
    features = np.ones((n, 1))
    n_nodes, feat_dim = features.shape

    sm_fea_s = sp.csr_matrix(features).toarray() 

    print('Laplacian Smoothing...') 
    for a in adj_norm_s:
        sm_fea_s = a.dot(sm_fea_s)

    print("sm_fea_s shape", sm_fea_s.shape)

    if cpu:
        adj_1st = torch.eye(n) + adj
        adj_label = torch.FloatTensor(adj_1st)
    else:
        adj_1st = (adj.to(device) + torch.eye(n).to(device))
        adj_label = adj_1st.reshape(-1)

    layers = args.num_linear_layers
    dims = [feat_dim] + args.age_dims 
    print("Model Dims", dims)

    sm_fea_s = torch.FloatTensor(sm_fea_s)
    adj_label = adj_label.reshape([-1,])
    print("sm_fea_s shape", sm_fea_s.shape)
    print("adj_label shape", adj_label.shape)

    if cpu:
        model = LinTrans(layers, dims)
        emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels)
        predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                                  args.num_gnn_layers, args.dropout)
        inx = sm_fea_s
    else:
        model = LinTrans(layers, dims).to(device)
        emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
        predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                                  args.num_gnn_layers, args.dropout).to(device)
        inx = sm_fea_s.cuda()
        adj_label = adj_label.cuda()

    pos_num = n_train_edges 
    neg_num = n_nodes*n_nodes-pos_num
    print("Number Pos Training Edges", pos_num)
    print("Number Neg Training Edges", neg_num)
    
    up_eta = (args.upth_ed - args.upth_st) / (args.epochs/args.upd)
    low_eta = (args.lowth_ed - args.lowth_st) / (args.epochs/args.upd)

    pos_inds, neg_inds = update_similarity(normalize(sm_fea_s.numpy()), args.upth_st, args.lowth_st, pos_num, neg_num)
    print("pos_inds shape", pos_inds.shape)
    print("neg_inds shape", neg_inds.shape)

    upth, lowth = update_threshold(args.upth_st, args.lowth_st, up_eta, low_eta)
    if cpu:
        pos_inds_cuda = torch.LongTensor(pos_inds)
    else:
        pos_inds_cuda = torch.LongTensor(pos_inds).cuda()

    print("model parameters {}".format(sum(p.numel() for p in model.parameters())))
    #print("predictor parameters {}".format(sum(p.numel() for p in predictor.parameters())))
    #print("total parameters {}".format(data.num_nodes*args.hidden_channels + 
    #sum(p.numel() for p in model.parameters())+sum(p.numel() for p in predictor.parameters())))
    evaluator = Evaluator(name='ogbl-ddi')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@20': Logger(args.runs, args),
        'Hits@30': Logger(args.runs, args),
    }

    for run in range(args.runs):
        torch.nn.init.xavier_uniform_(emb.weight)
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, inx, pos_inds_cuda, neg_inds, adj, split_edge,
                         optimizer, args.batch_size, args.cpu)
            print("epoch: ", epoch, " loss: ", loss)

            if epoch % args.eval_steps == 0:
                results = test(model, predictor, emb.weight, data.adj, split_edge,
                               evaluator, args.batch_size)
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---')

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


if __name__ == "__main__":
    main()

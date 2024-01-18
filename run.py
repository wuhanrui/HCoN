import numpy as np
from utils import load_data, dotdict, seed_everything, accuracy, normalize_sparse_hypergraph_symmetric
from model import HCoN
import torch
from torch import optim
import torch.nn.functional as F
import os
import argparse


"""
run: python run.py --gpu_id 0 --dataname citeseer
"""


def training(data, args, s = 2021):

    seed_everything(seed = s)

    H_trainX = torch.from_numpy(data.H_trainX.toarray()).float().cuda()
    X = torch.from_numpy(data.X.toarray()).float().cuda()
    Y = torch.from_numpy(data.Y).float().cuda()
    
    hx1 = torch.from_numpy(data.hx1.toarray()).float().cuda()
    hx2 = torch.from_numpy(data.hx2.toarray()).float().cuda()
    hy1 = torch.from_numpy(data.hy1.toarray()).float().cuda()
    hy2 = torch.from_numpy(data.hy2.toarray()).float().cuda()
    
    idx_train = torch.LongTensor(data.idx_train).cuda()
    idx_test = torch.LongTensor(data.idx_test).cuda()
    labels = torch.LongTensor(np.where(data.labels)[1]).cuda()

    gamma = args.gamma
    epochs = args.epochs
    learning_rate = args.learning_rate

    x_n_nodes = X.shape[0]
    y_n_nodes = Y.shape[0]
    pos_weight = float(H_trainX.shape[0] * H_trainX.shape[0] - H_trainX.sum()) / H_trainX.sum()
    
    model = HCoN(X.shape[1], Y.shape[1], args.dim_hidden, data.n_class)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    
    cost_val = []
    for epoch in range(epochs):
        model.train()

        recovered, x_output = model(hx1, hx2, X, hy1, hy2, Y, args.alpha, args.beta) 
        loss1 = F.nll_loss(x_output[idx_train], labels[idx_train])
        loss2 = F.binary_cross_entropy_with_logits(recovered, H_trainX, pos_weight=pos_weight)
        loss_train = loss1 + gamma * loss2
        
        acc_train = accuracy(x_output[idx_train], labels[idx_train])
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
        
#         loss_val = F.nll_loss(x_output[idx_val], labels[idx_val])
#         cost_val.append(loss_val.item())
#         acc_val = accuracy(x_output[idx_val], labels[idx_val])
        
        
#         if epoch > args.early_stop and cost_val[-1] > np.mean(cost_val[-(args.early_stop+1):-1]):
#             print("Early stopping...")
#             break
        
    # Test
    with torch.no_grad():
        model.eval()
        recovered, x_output = model(hx1, hx2, X, hy1, hy2, Y, args.alpha, args.beta) 
        loss_test = F.nll_loss(x_output[idx_test], labels[idx_test])
        acc_test = accuracy(x_output[idx_test], labels[idx_test])
        
    return acc_test.item()



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Hypergraph Collaborative Network (HCoN)')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--dataname', type=str, nargs='?', default='citeseer', help="dataset to run")
    setting = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = setting.gpu_id
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.cuda.current_device()
    
    H, X, Y, labels, idx_train_list, idx_test_list = load_data(setting.dataname)
        
    H_trainX = H.copy()
    Y = np.eye(H.shape[1])  # use identity matrix
    hx1, hx2 = normalize_sparse_hypergraph_symmetric(H_trainX)
    hy1, hy2 = normalize_sparse_hypergraph_symmetric(H_trainX.transpose())

    
    if setting.dataname == "citeseer":
        dim_hidden = 512
        learning_rate = 0.001
        weight_decay = 0.001
        gamma = 10
        alpha = 0.8
        beta = 0.2
    
    
    epochs = 200
    seed = 2021
    early = 100
        
    acc_test = []
    for trial in range(idx_train_list.shape[0]):
        idx_train = idx_train_list[trial]
        idx_test = idx_test_list[trial]
    
        data = dotdict()
        args = dotdict()

        data.X = X
        data.Y = Y
        data.H_trainX = H_trainX
        data.hx1 = hx1
        data.hx2 = hx2
        data.hy1 = hy1
        data.hy2 = hy2
        data.labels = labels
        data.idx_train = idx_train
        data.idx_test = idx_test
        data.n_class = labels.shape[1]

        args.dim_hidden = dim_hidden
        args.weight_decay = weight_decay
        args.epochs = epochs
        args.early_stop = early
        args.learning_rate = learning_rate
        args.gamma = gamma
        args.alpha = alpha
        args.beta = beta

        test = training(data, args, s=seed)
        acc_test.append(test)
        
        print(f'Trial: {trial+1}, Accuracy: {test}')

        
    acc_test = np.array(acc_test) * 100
    m_acc = np.mean(acc_test)
    s_acc = np.std(acc_test)
    print("Average accuracy: {:.4f}({:.4f})".format(m_acc, s_acc))


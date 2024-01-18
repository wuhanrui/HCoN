import torch
from torch import nn, optim
import torch.nn as nn
import torch.nn.functional as F
from layer import GraphConvolution
        

class HCoN(nn.Module):
    def __init__(self, input_feat_x_dim, input_feat_y_dim, dim_l1, nclass):
        super(HCoN, self).__init__()
        
        self.gcx1_part1 = GraphConvolution(input_feat_x_dim, dim_l1)
        self.gcx1_part2 = GraphConvolution(input_feat_y_dim, dim_l1)
        self.gcx2_part1 = GraphConvolution(dim_l1, nclass)
        self.gcx2_part2 = GraphConvolution(dim_l1, nclass)
        
        self.gcy1_part1 = GraphConvolution(input_feat_y_dim, dim_l1)
        self.gcy1_part2 = GraphConvolution(input_feat_x_dim, dim_l1)
        self.gcy2_part1 = GraphConvolution(dim_l1, nclass)
        self.gcy2_part2 = GraphConvolution(dim_l1, nclass)

    
    def forward(self, hx1, hx2, x0, hy1, hy2, y0, alpha, beta):    
        neg_slope = 0.2
        
        
        # layer 1
        x_part1 = self.gcx1_part1(x0, hx1) * alpha
        x_part2 = self.gcx1_part2(y0, hx2) * (1 - alpha)
        x1 = x_part1 + x_part2   
        x1 = F.leaky_relu(x1, negative_slope=neg_slope)

        
        y_part1 = self.gcy1_part1(y0, hy1) * beta
        y_part2 = self.gcy1_part2(x0, hy2) * (1 - beta)
        y1 = y_part1 + y_part2
        y1 = F.leaky_relu(y1, negative_slope=neg_slope)
        
        
        # layer 2        
        x_part1 = self.gcx2_part1(x1, hx1) * alpha
        x_part2 = self.gcx2_part2(y1, hx2) * (1 - alpha)
        x2 = x_part1 + x_part2 
        x2 = F.leaky_relu(x2, negative_slope=neg_slope)

        
        y_part1 = self.gcy2_part1(y1, hy1) * beta
        y_part2 = self.gcy2_part2(x1, hy2) * (1 - beta)
        y2 = y_part1 + y_part2
        y2 = F.leaky_relu(y2, negative_slope=neg_slope)  
        
        
        h_hat = torch.mm(x2, y2.t())
        h_hat = torch.sigmoid(h_hat)
        
        x_output = F.log_softmax(x2, dim=1)
        y_output = torch.sigmoid(y_part1 + y_part2)   
        
        
        return h_hat, x_output



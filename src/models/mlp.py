# src/models/mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim,dropout_rate=0.0):
        super().__init__()

        layers = []
        current_dim = input_dim

        # 1. 构建隐藏层模块 Dynamic
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = h_dim
        
        # 2. 添加最终的输出层
        # 考虑到为分类任务，输出层不需激活函数，使用Loss函数处理
        layers.append(nn.Linear(current_dim, output_dim))

        # 3. 封装进 nn.Sequential
        # layers为list，需将其解包
        self.net = nn.Sequential(*layers)
    
    def forward(self, x:torch.Tensor):
        """
        前向传播
        """
        x = x.view(x.size(0),-1)

        return self.net(x)
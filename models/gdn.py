import torch
import torch.nn as nn
import torch.nn.functional as F

class GDN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_time_windows):
        super(GDN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.num_time_windows = num_time_windows

    def forward(self, x):  # 确保 forward 函数接收一个输入参数 x
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # 将输出通过 sigmoid 转换为概率
        return x

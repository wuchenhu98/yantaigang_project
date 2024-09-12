import torch
import torch.nn as nn
import torch.nn.functional as F

class GDN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_time_windows):
        super(GDN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.num_time_windows = num_time_windows
        
        # 用于存储每个特征的历史均值和标准差
        self.feature_means = nn.Parameter(torch.zeros(input_dim), requires_grad=False)
        self.feature_stds = nn.Parameter(torch.ones(input_dim), requires_grad=False)

    def forward(self, x):
        # 计算每个特征与均值的距离，并归一化到标准差
        normalized_distances = torch.abs((x - self.feature_means) / (self.feature_stds + 1e-6))
        anomaly_scores = torch.sigmoid(normalized_distances)  # 将距离映射到 [0, 1] 作为异常得分
        
        # 正常的前向传播，故障预测
        x = F.relu(self.fc1(x))
        predictions = torch.sigmoid(self.fc2(x))  # 将输出通过 sigmoid 转换为概率
        
        return predictions, anomaly_scores

    # 更新均值和方差
    def update_statistics(self, data):
        with torch.no_grad():
            self.feature_means.data = data.mean(dim=0)
            self.feature_stds.data = data.std(dim=0) + 1e-6  # 避免除以0

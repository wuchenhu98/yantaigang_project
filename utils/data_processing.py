import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch

def load_device_data(file_path):
    device_data = pd.read_csv(file_path)
    features = device_data[["车速", "发动机转速", "油耗", "故障码数量", "发动机扭矩", "摩擦扭矩", "燃料流量", 
                           "空气进气量", "SCR上游NOX传感器输出值", "SCR下游NOX传感器输出值", 
                           "SCR入口温度", "SCR出口温度", "水温"]]
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return torch.tensor(features_scaled, dtype=torch.float32)

def load_maintenance_data(file_path):
    maintenance_data = pd.read_csv(file_path)
    # 转换维修日志数据为one-hot编码 (假设你需要这样的处理)
    maintenance_data_onehot = pd.get_dummies(maintenance_data["repair_type"])
    return maintenance_data_onehot, maintenance_data_onehot.columns.tolist()  # 返回 DataFrame


def generate_time_window_labels(maintenance_data_df, num_time_windows):
    time_window_labels = []
    for i in range(1, num_time_windows + 1):
        # 使用 shift 方法，填充缺失值并将数据类型转换为 float
        window_label = maintenance_data_df.shift(-i).fillna(0).astype(float)
        time_window_labels.append(window_label)

    # 将每个时间窗口的标签转换为 Tensor
    time_window_labels = [torch.tensor(label.values, dtype=torch.float32) for label in time_window_labels]
    
    return torch.stack(time_window_labels, dim=1)  # 堆叠成 [batch_size, num_time_windows, num_labels]



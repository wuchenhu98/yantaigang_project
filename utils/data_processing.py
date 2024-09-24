import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_device_data(file_path, num_time_windows):
    device_data = pd.read_csv(file_path)
    features = device_data[["车速", "发动机转速", "油耗", "故障码数量", "发动机扭矩", "摩擦扭矩", "燃料流量", 
                           "空气进气量", "SCR上游NOX传感器输出值", "SCR下游NOX传感器输出值", 
                           "SCR入口温度", "SCR出口温度", "水温"]]
    
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 创建时间特征
    time_steps = np.arange(len(features))
    time_features = np.tile(np.arange(1, num_time_windows + 1), (len(features), 1))

    # 将时间步数和时间窗口长度作为新的特征
    features_with_time = np.hstack((features_scaled, time_steps[:, None], time_features))
    
    return torch.tensor(features_with_time, dtype=torch.float32)

def load_maintenance_data(file_path):
    maintenance_data = pd.read_csv(file_path)
    maintenance_data_onehot = pd.get_dummies(maintenance_data["repair_type"])
    return maintenance_data_onehot, maintenance_data_onehot.columns.tolist()

def generate_time_window_labels(maintenance_data_df, num_time_windows):
    time_window_labels = []
    for i in range(1, num_time_windows + 1):
        window_label = maintenance_data_df.shift(-i).fillna(0).astype(float)
        time_window_labels.append(window_label)
    time_window_labels = [torch.tensor(label.values, dtype=torch.float32) for label in time_window_labels]
    
    return torch.stack(time_window_labels, dim=1)

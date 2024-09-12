import torch
import csv
import pandas as pd  # 加载 pandas 以处理 DataFrame
from models.gdn import GDN
from utils.data_processing import load_device_data, load_maintenance_data
import os
from datetime import datetime

device_data_file = 'data/device_operation_data.csv'
maintenance_data_file = 'data/maintenance_logs.csv'
model_file = 'models/gdn_model.pth'

# 获取当前系统时间并格式化为指定的文件名格式
current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)  # 如果目录不存在则创建
output_file = os.path.join(output_dir, f'predictions_{current_time}.csv')

# 加载设备运行数据，包括时间戳
device_data_df = pd.read_csv(device_data_file)
device_data = load_device_data(device_data_file)
device_data = torch.as_tensor(device_data, dtype=torch.float32)

# 从维修日志中提取故障种类和维修数据
maintenance_data, fault_types = load_maintenance_data(maintenance_data_file)

# 获取特征列名，用于异常得分
feature_names = ["车速", "发动机转速", "油耗", "故障码数量", "发动机扭矩", "摩擦扭矩", "燃料流量", 
                 "空气进气量", "SCR上游NOX传感器输出值", "SCR下游NOX传感器输出值", 
                 "SCR入口温度", "SCR出口温度", "水温"]

# 定义模型参数
input_dim = device_data.shape[1]
hidden_dim = 64
output_dim = len(fault_types)  # 确保输出维度与训练时一致
num_time_windows = 6  # 预测未来1-6周

# 加载训练好的模型
model = GDN(input_dim, hidden_dim, output_dim, num_time_windows)
model.load_state_dict(torch.load(model_file))
model.eval()

# 在运行模型前，先使用设备数据更新特征均值和标准差
model.update_statistics(device_data)

# 时间窗口的描述（1到6周）
time_window_descriptions = ["一周内", "两周内", "三周内", "四周内", "五周内", "六周内"]

# 对设备运行数据进行预测
predictions, anomaly_scores = [], []
with torch.no_grad():
    pred, anomaly_score = model(device_data)  # 获取预测和异常得分
    predictions.append(pred)
    anomaly_scores.append(anomaly_score)

# 将预测结果保存到 CSV 文件
with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 写入表头，使用故障种类替代 Prediction 1, Prediction 2 等，并增加异常得分列
    anomaly_score_headers = [f"{name} 异常得分" for name in feature_names]
    header = ['Time Window', 'Timestamp'] + fault_types + anomaly_score_headers
    writer.writerow(header)
    
    # 写入每个时间窗口的预测结果
    timestamps = device_data_df['timestamp']  # 从 DataFrame 中提取时间戳
    num_devices = device_data.shape[0]  # 设备数量
    
    for i, time_window in enumerate(time_window_descriptions):  # 遍历每个时间窗口
        for idx in range(num_devices):  # 遍历每个设备
            timestamp = timestamps.iloc[idx]  # 根据索引获取时间戳
            
            # 获取当前设备在当前时间窗口的预测故障概率和异常得分
            prediction_row = predictions[0][idx].tolist()  # 获取当前时间窗口的预测
            anomaly_score_row = anomaly_scores[0][idx].tolist()  # 获取当前设备的各维度异常得分
            
            # 将预测结果和异常得分合并，并保留3位小数
            combined_row = [time_window, timestamp] + [f"{x:.3f}" for x in prediction_row] + [f"{x:.3f}" for x in anomaly_score_row]
            writer.writerow(combined_row)

print(f'预测结果已保存到 {output_file}')

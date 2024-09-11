import torch
import csv
import pandas as pd  # 加载 pandas 以处理 DataFrame
from models.gdn import GDN
from utils.data_processing import load_device_data, load_maintenance_data

device_data_file = 'data/device_operation_data.csv'
maintenance_data_file = 'data/maintenance_logs.csv'
model_file = 'models/gdn_model.pth'
output_file = 'predictions.csv'

# 加载设备运行数据，包括时间戳
device_data_df = pd.read_csv(device_data_file)
device_data = load_device_data(device_data_file)
device_data = torch.as_tensor(device_data, dtype=torch.float32)

# 从维修日志中提取故障种类和维修数据
maintenance_data, fault_types = load_maintenance_data(maintenance_data_file)

# 定义模型参数
input_dim = device_data.shape[1]
hidden_dim = 64
output_dim = len(fault_types)  # 确保输出维度与训练时一致
num_time_windows = 6  # 预测未来1-6周

# 加载训练好的模型
model = GDN(input_dim, hidden_dim, output_dim, num_time_windows)
model.load_state_dict(torch.load(model_file))
model.eval()

# 时间窗口的描述（1到6周）
time_window_descriptions = ["一周内", "两周内", "三周内", "四周内", "五周内", "六周内"]

# 对设备运行数据进行预测
predictions = []
with torch.no_grad():
    for i in range(num_time_windows):  # 遍历6个时间窗口
        pred = model(device_data)
        predictions.append(pred)

# 将预测结果保存到 CSV 文件
with open(output_file, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 写入表头，使用故障种类替代 Prediction 1, Prediction 2 等
    header = ['Time Window', 'Timestamp'] + fault_types
    writer.writerow(header)
    
    # 写入每个时间窗口的预测结果
    timestamps = device_data_df['timestamp']  # 从 DataFrame 中提取时间戳
    for i, time_window in enumerate(time_window_descriptions):  # 遍历每个时间窗口
        for idx, row in enumerate(predictions[i]):
            timestamp = timestamps.iloc[idx]  # 根据索引获取时间戳
            prediction_row = [time_window, timestamp] + row.tolist()
            writer.writerow(prediction_row)

print(f'预测结果已保存到 {output_file}')

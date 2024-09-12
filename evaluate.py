import torch
import pandas as pd
from models.gdn import GDN
from utils.data_processing import load_device_data, load_maintenance_data
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from datetime import datetime
import os

# 定义输出文件路径
predictions_dir = 'results/predictions'
anomaly_scores_dir = 'results/anomaly_scores'
os.makedirs(predictions_dir, exist_ok=True)
os.makedirs(anomaly_scores_dir, exist_ok=True)

# 生成带时间戳的文件名
timestamp_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
predictions_file = os.path.join(predictions_dir, f'predictions_{timestamp_str}.xlsx')
anomaly_scores_file = os.path.join(anomaly_scores_dir, f'anomaly_scores_{timestamp_str}.xlsx')

# 加载设备运行数据，包括时间戳
device_data_df = pd.read_csv('data/device_operation_data.csv')
device_data = load_device_data('data/device_operation_data.csv')
device_data = torch.as_tensor(device_data, dtype=torch.float32)

# 从维修日志中提取故障种类和维修数据
maintenance_data, fault_types = load_maintenance_data('data/maintenance_logs.csv')

# 定义模型参数
input_dim = device_data.shape[1]
hidden_dim = 64
output_dim = len(fault_types)  # 确保输出维度与训练时一致
num_time_windows = 6  # 预测未来1-6周

# 加载训练好的模型
model = GDN(input_dim, hidden_dim, output_dim, num_time_windows)
model.load_state_dict(torch.load('models/gdn_model.pth'))
model.eval()

# 时间窗口的描述（1到6周）
time_window_descriptions = ["一周内", "两周内", "三周内", "四周内", "五周内", "六周内"]

# 对设备运行数据进行预测
predictions = []
anomaly_scores = []
with torch.no_grad():
    for i in range(num_time_windows):  # 遍历6个时间窗口
        pred, anomaly_score = model(device_data)
        predictions.append(pred)
        anomaly_scores.append(anomaly_score)

# 将预测结果保存到 Excel 文件
predictions_rows = []
timestamps = device_data_df['timestamp']  # 从 DataFrame 中提取时间戳
for i, time_window in enumerate(time_window_descriptions):  # 遍历每个时间窗口
    for idx, row in enumerate(predictions[i]):
        timestamp = timestamps.iloc[idx]  # 根据索引获取时间戳
        # 获取当前时间戳下所有维度中异常得分最高的3个维度
        top_anomalies = anomaly_scores[i][idx].topk(3).indices.tolist()
        top_anomaly_features = [device_data_df.columns[1:][j] for j in top_anomalies]  # 获取维度名称
        top_anomaly_str = ', '.join(top_anomaly_features)

        # 将每个故障概率转换为百分数并保留3位小数
        prediction_row = [time_window, timestamp] + [f"{x * 100:.3f}%" for x in row.tolist()] + [top_anomaly_str]
        predictions_rows.append(prediction_row)

# 更新故障列名，添加“概率”后缀
fault_types_with_suffix = [f"{fault}概率" for fault in fault_types]

# 将预测结果保存为 DataFrame
predictions_df = pd.DataFrame(predictions_rows, columns=['Time Window', 'Timestamp'] + fault_types_with_suffix + ['故障发生根因'])

# 保存为 Excel 文件并调整列宽
with pd.ExcelWriter(predictions_file, engine='openpyxl') as writer:
    predictions_df.to_excel(writer, index=False, sheet_name='Predictions')
    worksheet = writer.sheets['Predictions']
    
    # 自动调整列宽
    for column_cells in worksheet.columns:
        max_length = max(len(str(cell.value)) for cell in column_cells)
        worksheet.column_dimensions[column_cells[0].column_letter].width = max_length + 2

print(f'预测结果已保存到 {predictions_file}')

# 保存异常得分到 Excel 文件
anomaly_score_rows = []
for idx in range(device_data.shape[0]):
    timestamp = timestamps.iloc[idx]  # 根据索引获取时间戳
    for i, time_window in enumerate(time_window_descriptions):  # 遍历每个时间窗口
        anomaly_score_row = [timestamp] + anomaly_scores[i][idx].tolist()  # 当前设备的异常得分
        anomaly_score_rows.append(anomaly_score_row)

# 生成异常得分的 DataFrame，移除 '异常' 列
anomaly_score_columns = ['Timestamp'] + device_data_df.columns[1:-1].tolist()  # 去掉 '异常' 列

# 创建异常得分 DataFrame
anomaly_scores_df = pd.DataFrame(anomaly_score_rows, columns=anomaly_score_columns)

# 保存为 Excel 文件并调整列宽
with pd.ExcelWriter(anomaly_scores_file, engine='openpyxl') as writer:
    anomaly_scores_df.to_excel(writer, index=False, sheet_name='Anomaly Scores')
    worksheet = writer.sheets['Anomaly Scores']
    
    # 自动调整列宽
    for column_cells in worksheet.columns:
        max_length = max(len(str(cell.value)) for cell in column_cells)
        worksheet.column_dimensions[column_cells[0].column_letter].width = max_length + 2

print(f'异常得分已保存到 {anomaly_scores_file}')

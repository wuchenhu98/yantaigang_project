import torch
import pandas as pd
from models.gdn import GDN
from utils.data_processing import load_device_data, load_maintenance_data
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from datetime import datetime
import os

# 定义输出文件路径
local_predictions_dir = 'results/local_prediction'
anomaly_scores_dir = 'results/anomaly_scores'
overall_prediction_dir = 'results/overall_prediction'
os.makedirs(local_predictions_dir, exist_ok=True)
os.makedirs(anomaly_scores_dir, exist_ok=True)
os.makedirs(overall_prediction_dir, exist_ok=True)

# 生成带时间戳的文件名
timestamp_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
local_predictions_file = os.path.join(local_predictions_dir, f'local_prediction_{timestamp_str}.xlsx')
anomaly_scores_file = os.path.join(anomaly_scores_dir, f'anomaly_scores_{timestamp_str}.xlsx')
overall_prediction_file = os.path.join(overall_prediction_dir, f'overall_prediction_{timestamp_str}.xlsx')

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

# 将本地预测结果保存到 Excel 文件
predictions_rows = []
timestamps = device_data_df['timestamp']  # 从 DataFrame 中提取时间戳
for idx in range(device_data.shape[0]):  # 遍历每个时间戳
    timestamp = timestamps.iloc[idx]  # 根据索引获取时间戳
    for i, time_window in enumerate(time_window_descriptions):  # 遍历每个时间窗口
        # 获取当前时间戳下所有维度中异常得分最高的3个维度
        top_anomalies = anomaly_scores[i][idx].topk(3).indices.tolist()
        top_anomaly_features = [device_data_df.columns[1:][j] for j in top_anomalies]  # 获取维度名称
        top_anomaly_str = ', '.join(top_anomaly_features)

        # 将每个故障概率转换为百分数并保留3位小数
        prediction_row = [timestamp, time_window] + [f"{x * 100:.3f}%" for x in predictions[i][idx].tolist()] + [top_anomaly_str]
        predictions_rows.append(prediction_row)

# 更新故障列名，添加“概率”后缀
fault_types_with_suffix = [f"{fault}概率" for fault in fault_types]

# 将预测结果保存为 DataFrame，时间戳列作为第一列，时间窗口列名为“未来时间窗口”
predictions_df = pd.DataFrame(predictions_rows, columns=['时间', '未来时间窗口'] + fault_types_with_suffix + ['故障发生根因'])

# 保存为 Excel 文件并调整列宽
with pd.ExcelWriter(local_predictions_file, engine='openpyxl') as writer:
    predictions_df.to_excel(writer, index=False, sheet_name='Predictions')
    worksheet = writer.sheets['Predictions']
    
    # 自动调整列宽
    for column_cells in worksheet.columns:
        max_length = max(len(str(cell.value)) for cell in column_cells)
        worksheet.column_dimensions[column_cells[0].column_letter].width = max_length + 2

print(f'本地预测结果已保存到 {local_predictions_file}')

# 保存异常得分到 Excel 文件
anomaly_score_rows = []
for idx in range(device_data.shape[0]):
    timestamp = timestamps.iloc[idx]  # 根据索引获取时间戳
    for i, time_window in enumerate(time_window_descriptions):  # 遍历每个时间窗口
        anomaly_score_row = [timestamp, time_window] + anomaly_scores[i][idx].tolist()  # 当前设备的异常得分，并添加时间窗口
        anomaly_score_rows.append(anomaly_score_row)

# 生成异常得分的 DataFrame，移除 '异常' 列，时间戳列作为第一列，时间窗口列名为“未来时间窗口”
anomaly_score_columns = ['时间', '未来时间窗口'] + device_data_df.columns[1:].tolist()  # 确保列数和数据匹配


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

# 计算每个时间窗口的整体故障概率预测，并按故障概率排序
overall_predictions = []
overall_causes = []  # 用于存储每个时间窗口的主要故障原因
for i in range(num_time_windows):
    overall_prediction = torch.mean(predictions[i], dim=0)  # 计算该时间窗口的平均概率
    overall_predictions.append(overall_prediction)

    # 计算整体异常得分
    overall_anomalies = torch.mean(anomaly_scores[i], dim=0)  # 获取该时间窗口的平均异常得分
    top_anomalies_indices = overall_anomalies.topk(3).indices.tolist()  # 获取最高的三个异常维度索引
    top_anomaly_features = [device_data_df.columns[1:][j] for j in top_anomalies_indices]  # 获取维度名称
    top_anomaly_str = ', '.join(top_anomaly_features)
    overall_causes.append(top_anomaly_str)  # 存储每个时间窗口的主要故障原因

# 保存整体故障预测为 Excel 文件，按时间窗口分别保存
overall_prediction_rows = []
for i, time_window in enumerate(time_window_descriptions):
    overall_predictions_sorted, sorted_indices = torch.sort(overall_predictions[i], descending=True)  # 按概率排序
    sorted_fault_types = [fault_types[j] for j in sorted_indices.tolist()]  # 获取排序后的故障类型
    overall_prediction_rows += [[time_window, f"{sorted_fault_types[k]}概率", f"{overall_predictions_sorted[k] * 100:.3f}%", overall_causes[i]]
                                for k in range(len(sorted_fault_types))]

# 创建整体预测 DataFrame，按时间窗口分别列出，并添加“故障发生原因”列
overall_predictions_df = pd.DataFrame(overall_prediction_rows, columns=['未来时间窗口', '故障类型', '预测概率', '故障发生原因'])

# 保存整体预测结果为 Excel 文件并调整列宽
with pd.ExcelWriter(overall_prediction_file, engine='openpyxl') as writer:
    overall_predictions_df.to_excel(writer, index=False, sheet_name='Overall Prediction')
    worksheet = writer.sheets['Overall Prediction']
    
    # 自动调整列宽
    for column_cells in worksheet.columns:
        max_length = max(len(str(cell.value)) for cell in column_cells)
        worksheet.column_dimensions[column_cells[0].column_letter].width = max_length + 2

print(f'整体预测结果已保存到 {overall_prediction_file}')

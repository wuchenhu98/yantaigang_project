import torch
import torch.optim as optim
from models.gdn import GDN
from utils.data_processing import load_device_data, load_maintenance_data, generate_time_window_labels

device_data_file = 'data/device_operation_data.csv'
maintenance_data_file = 'data/maintenance_logs.csv'

# 加载数据
device_data = load_device_data(device_data_file)
maintenance_data_df, fault_types = load_maintenance_data(maintenance_data_file)  # 返回 DataFrame

# 假设我们每10条设备数据对应一条维护记录
num_segments = len(maintenance_data_df)
device_data = device_data.view(num_segments, -1, device_data.shape[1]).mean(dim=1)

# 为每个时间窗口生成标签
num_time_windows = 6  # 预测未来1-6周的维修事件
maintenance_labels = generate_time_window_labels(maintenance_data_df, num_time_windows)  # 使用 DataFrame 生成标签

# 定义模型
input_dim = device_data.shape[1]
output_dim = len(fault_types)  # 确保 output_dim 为维修故障种类的数量
hidden_dim = 64

model = GDN(input_dim, hidden_dim, output_dim, num_time_windows)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 更新特征的均值和方差
model.update_statistics(device_data)

# 训练模型
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    # 模型输出为多个时间窗口的预测和异常得分
    predictions, anomaly_scores = model(device_data)
    
    # 计算每个时间窗口的损失并累加
    total_loss = 0
    for i in range(num_time_windows):
        loss = torch.nn.functional.mse_loss(predictions, maintenance_labels[:, i, :])
        total_loss += loss
    
    total_loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4f}')
    # 打印每个维度的平均异常得分
    print(f'Anomaly scores for each dimension: {anomaly_scores.mean(dim=0).detach().numpy()}')

# 保存模型
save_path = 'models/gdn_model.pth'
torch.save(model.state_dict(), save_path)

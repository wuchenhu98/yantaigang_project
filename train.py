import torch
import torch.optim as optim
import os
from sklearn.metrics import f1_score, recall_score, accuracy_score
from models.gdn import GDN
from utils.data_processing import load_device_data, load_maintenance_data, generate_time_window_labels

device_data_file = 'data/device_operation_data.csv'
maintenance_data_file = 'data/maintenance_logs.csv'

# 加载数据
device_data = load_device_data(device_data_file)
maintenance_data_df, fault_types = load_maintenance_data(maintenance_data_file)

num_segments = len(maintenance_data_df)
device_data = device_data.view(num_segments, -1, device_data.shape[1]).mean(dim=1)

num_time_windows = 6
maintenance_labels = generate_time_window_labels(maintenance_data_df, num_time_windows)

input_dim = device_data.shape[1]
output_dim = len(fault_types)
hidden_dim = 64

model = GDN(input_dim, hidden_dim, output_dim, num_time_windows)
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.update_statistics(device_data)

epochs = 50
all_labels = []
all_preds = []
all_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions, anomaly_scores = model(device_data)
    
    total_loss = 0
    for i in range(num_time_windows):
        loss = torch.nn.functional.mse_loss(predictions, maintenance_labels[:, i, :])
        total_loss += loss
    
    total_loss.backward()
    optimizer.step()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4f}')
    
    all_labels.append(maintenance_labels[:, i, :].cpu().detach().numpy())
    all_preds.append(predictions.cpu().detach().numpy())
    all_losses.append(total_loss.item())

save_path = 'models/gdn_model.pth'
if os.path.exists(save_path):
    os.remove(save_path)
torch.save(model.state_dict(), save_path)
print(f'模型已保存到 {save_path}')

# 输出F1得分，准确率，召回率
labels_concat = torch.cat([torch.tensor(l) for l in all_labels])
preds_concat = torch.cat([torch.tensor(p) for p in all_preds])

# 修改F1, 召回率，准确率的计算，设置zero_division参数避免警告
f1 = f1_score(labels_concat, preds_concat > 0.5, average='weighted', zero_division=1)
recall = recall_score(labels_concat, preds_concat > 0.5, average='weighted', zero_division=1)
accuracy = accuracy_score(labels_concat, preds_concat > 0.5)

print(f'F1得分: {f1:.4f}, 准确率: {accuracy:.4f}, 召回率: {recall:.4f}')
print(f'训练中的平均损失: {sum(all_losses) / epochs:.4f}')

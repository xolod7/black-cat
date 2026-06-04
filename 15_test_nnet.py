# test_nnet.py
"""
Тест: многослойная нейронная сеть (PyTorch).
Архитектура с bottleneck для выделения информативных признаков.
Поддерживает GPU.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import datetime as dt
import im2data
import test_utils

# ===================== НАСТРОЙКИ =====================
fileIm = test_utils.get_image_path("1.png")
n = 240000
rm = 2             # общее число признаков Раунд 1
# rm = 10            # общее число признаков Раунд 2,3
# rm = 500           # общее число признаков Раунд 4,5  
rotate = False     # поворот пространства признаков Раунд 1,2,4
# rotate = True     # поворот пространства признаков Раунд 3,5

use_gpu = True
device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

# Гиперпараметры
epochs = 300
batch_size = 4096
learning_rate = 1e-3
weight_decay = 1e-5
scheduler_patience = 20
scheduler_factor = 0.5
# =====================================================

print(f"Device: {device}")
print(f"Загрузка данных: rm={rm}, rotate={rotate}")

im, X_train_full, X_test_new, X_train, y_train, X_test, y_test, rot_mat = \
    im2data.obraz2d2(fileIm, n, rm, rotate=rotate)

# Перенос на device
X_train_t = X_train.to(device)
y_train_t = y_train.to(device)
X_test_t = X_test.to(device)
y_test_t = y_test.to(device)


class RegressionNet(nn.Module):
    """
    Сеть с bottleneck: 
    входной слой -> сжатие до bottleneck_size -> расширение -> предсказание.
    
    Идея: bottleneck заставляет сеть выучить, какие признаки важны,
    сжав информацию из rm измерений в несколько.
    """
    def __init__(self, input_dim, bottleneck_size=4):
        super().__init__()
        
        # Адаптивный первый слой
        if input_dim <= 10:
            first_hidden = max(input_dim * 2, 8)
            dropout = 0.0
        elif input_dim <= 100:
            first_hidden = 64
            dropout = 0.05
        else:
            first_hidden = 64
            dropout = 0.1
        
        layers = [
            nn.Linear(input_dim, first_hidden),
            nn.BatchNorm1d(first_hidden),
            nn.GELU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        layers += [
            nn.Linear(first_hidden, bottleneck_size),
            nn.BatchNorm1d(bottleneck_size),
            nn.GELU(),
            
            nn.Linear(bottleneck_size, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            
            nn.Linear(32, 1)
        ]
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


# Создание модели
# Адаптивный bottleneck
bottleneck_size = 4 if rm > 10 else max(rm, 2)
model = RegressionNet(rm, bottleneck_size=bottleneck_size).to(device)
print(f"Параметров в модели: {sum(p.numel() for p in model.parameters()):,}")

# Оптимизатор и планировщик
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=scheduler_patience, factor=scheduler_factor,
    min_lr=1e-6
)
criterion = nn.MSELoss()

# DataLoader
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Обучение
print("Обучение нейросети...")
train_errors = []
val_errors = []
best_val_rmse = float('inf')
best_state = None

start = dt.datetime.now()

for epoch in range(epochs):
    # Train
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        # epoch_loss += loss.item() * xb.size(0)
    
    # train_rmse = np.sqrt(epoch_loss / len(train_dataset))
    
    model.eval()
    with torch.no_grad():
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        train_rmse = np.sqrt(loss.item())

    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_test_t)
        val_loss = criterion(val_pred, y_test_t)
        val_rmse = np.sqrt(val_loss.item())
    
    scheduler.step(val_rmse)
    
    train_errors.append(train_rmse)
    val_errors.append(val_rmse)
    
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    if (epoch + 1) % 10 == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f"Эпоха {epoch+1}/{epochs}: "
              f"train RMSE={train_rmse:.6f}, val RMSE={val_rmse:.6f}, lr={lr:.2e}")

train_time = dt.datetime.now() - start

# Восстановление лучшей модели
model.load_state_dict(best_state)
model.eval()

# Предсказания на тестовой выборке с новым шумом
def predict_torch(X_np):
    X_t = torch.from_numpy(X_np).float().to(device) if isinstance(X_np, np.ndarray) \
        else X_np.float().to(device)
    with torch.no_grad():
        return model(X_t).cpu().numpy()

start = dt.datetime.now()
y_pred_full = test_utils.predict_in_batches(predict_torch, X_test_new.numpy() 
    if isinstance(X_test_new, torch.Tensor) else X_test_new, batch_size=50000)
pred_time = dt.datetime.now() - start

rmse_full = test_utils.compute_rmse(y_pred_full, im.reshape(-1))

test_utils.print_results(f"Neural Network (PyTorch)", rmse_full, train_time, pred_time)
print(f"Лучший RMSE на валидации: {best_val_rmse:.6f}")

# Графики
test_utils.plot_learning_curve(train_errors, val_errors, "Нейросеть: кривая обучения")

# Визуализация — эталон и результат в одном окне
var = test_utils.show_comparison(
    im, y_pred_full, im.shape,
    method_name="Neural Network", rm=rm, rmse=rmse_full
)

test_utils.show_surface_3d(im, "Эталон")
test_utils.show_surface_3d(var, "Neural Network")
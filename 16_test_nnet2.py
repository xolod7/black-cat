# test_nnet.py
"""
Тест: многослойная нейронная сеть (PyTorch).
Два режима архитектуры: простая (малое rm) и с bottleneck (большое rm).
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
# rm = 1000           # общее число признаков Раунд 6
rotate = False     # поворот пространства признаков Раунд 1,2,4,6
# rotate = True     # поворот пространства признаков Раунд 3,5

# Гиперпараметры обучения
epochs = 100        # Раунд 1,2,3
# epochs = 300        # Раунд 4,5

use_gpu = True
device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

# Адаптивный batch_size:
# маленький batch = больше шагов = быстрее сходимость
# но слишком маленький плохо утилизирует GPU
if rm <= 20:
    batch_size = 256
elif rm <= 500:
    batch_size = 512
else:
    batch_size = 1024

learning_rate = 1e-3
weight_decay = 1e-5

# results_csv = "results.csv"
# =====================================================

print(f"Device: {device}")
print(f"Загрузка данных: rm={rm}, rotate={rotate}")

im, X_train_full, X_test_new, X_train, y_train, X_test, y_test, rot_mat = \
    im2data.obraz2d2(fileIm, n, rm, rotate=rotate)

X_train_t = X_train.to(device)
y_train_t = y_train.to(device)
X_test_t = X_test.to(device)
y_test_t = y_test.to(device)


class SimpleNet(nn.Module):
    """
    Простая сеть без BatchNorm/Dropout — для малого rm.
    Архитектура близка к оригинальному коду из документа.
    """
    def __init__(self, input_dim, bottleneck=3, hidden=50, n_hidden_layers=8):
        super().__init__()
        layers = [
            nn.Linear(input_dim, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, hidden),
            nn.ReLU(),
        ]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BottleneckNet(nn.Module):
    """
    Сеть с BatchNorm и Dropout — для большого rm,
    где нужно отфильтровать шумовые признаки.
    """
    def __init__(self, input_dim, bottleneck=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, bottleneck),
            nn.BatchNorm1d(bottleneck),
            nn.ReLU(),

            nn.Linear(bottleneck, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),

            nn.Linear(50, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),

            nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.net(x)


# Выбор архитектуры в зависимости от rm
if rm <= 20:
    bottleneck = max(rm, 3)
    model = SimpleNet(rm, bottleneck=bottleneck, hidden=50, n_hidden_layers=8).to(device)
    use_scheduler = False
else:
    bottleneck = 4
    model = BottleneckNet(rm, bottleneck=bottleneck).to(device)
    use_scheduler = True

n_params = sum(p.numel() for p in model.parameters())
print(f"Архитектура: {'SimpleNet' if rm <= 20 else 'BottleneckNet'}")
print(f"Параметров в модели: {n_params:,}")
print(f"Batch size: {batch_size}")
print(f"Шагов за эпоху: {n // batch_size}")

# Оптимизатор
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

if use_scheduler:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=30, factor=0.5, min_lr=1e-6
    )

criterion = nn.MSELoss()

# DataLoader
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Для быстрой валидации — один батч
val_batch_size = min(len(X_test_t), 30000)

# Обучение
print(f"\nОбучение нейросети ({epochs} эпох)...")
train_errors = []
val_errors = []
best_val_rmse = float('inf')
best_state = None

start_total = dt.datetime.now()

for epoch in range(epochs):
    start_epoch = dt.datetime.now()

    # Train
    model.train()
    epoch_loss = 0.0
    n_batches = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

    train_rmse = np.sqrt(epoch_loss / n_batches)

    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_test_t)
        val_loss = criterion(val_pred, y_test_t)
        val_rmse = np.sqrt(val_loss.item())

    if use_scheduler:
        scheduler.step(val_rmse)

    train_errors.append(train_rmse)
    val_errors.append(val_rmse)

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

    epoch_time = dt.datetime.now() - start_epoch
    if (epoch + 1) % 10 == 0 or epoch == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f"Эпоха {epoch+1:4d}/{epochs}: "
              f"train={train_rmse:.6f}, val={val_rmse:.6f}, "
              f"lr={lr:.1e}, {epoch_time}")

train_time = dt.datetime.now() - start_total

# Восстановление лучшей модели
model.load_state_dict(best_state)
model.eval()


# Предсказания
def predict_torch(X_input):
    if isinstance(X_input, np.ndarray):
        X_t = torch.from_numpy(X_input).float().to(device)
    else:
        X_t = X_input.float().to(device)
    with torch.no_grad():
        return model(X_t).cpu().numpy()


# На полном тестовом наборе с новым шумом
X_test_new_np = X_test_new.numpy() if isinstance(X_test_new, torch.Tensor) else X_test_new

start = dt.datetime.now()
y_pred_full = test_utils.predict_in_batches(predict_torch, X_test_new_np, batch_size=50000)
pred_time = dt.datetime.now() - start

rmse_full = test_utils.compute_rmse(y_pred_full, im.reshape(-1))

test_utils.print_results(f"Neural Network (PyTorch)", rmse_full, train_time, pred_time)
print(f"Лучший RMSE на валидации: {best_val_rmse:.6f}")
print(f"Batch size: {batch_size}")
print(f"Всего шагов оптимизации: {epochs * (n // batch_size)}")

# Графики
test_utils.plot_learning_curve(train_errors, val_errors, "Нейросеть: кривая обучения")

# Визуализация — эталон и результат в одном окне
var = test_utils.show_comparison(
    im, y_pred_full, im.shape,
    method_name="Neural Network", rm=rm, rmse=rmse_full
)

test_utils.show_surface_3d(im, "Эталон")
test_utils.show_surface_3d(var, "Neural Network")
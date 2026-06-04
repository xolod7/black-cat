# test_tabnet.py
"""
Тест: TabNet — нейросеть, специально разработанная для табличных данных.
Имеет встроенный механизм отбора признаков (attention).
Потенциально очень хороший кандидат для данной задачи.
pip install pytorch-tabnet
"""
import numpy as np
import torch
import datetime as dt
from pytorch_tabnet.tab_model import TabNetRegressor
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
device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

# Гиперпараметры TabNet
n_d = 32              # ширина decision слоёв
n_a = 32              # ширина attention слоёв
n_steps = 5           # количество шагов attention
gamma = 1.5           # коэффициент разреженности attention
lambda_sparse = 1e-3  # регуляризация разреженности
max_epochs = 150
patience = 30
batch_size = 4096
virtual_batch_size = 256
# =====================================================

print(f"Device: {device}")
print(f"Загрузка данных: rm={rm}, rotate={rotate}")

im, X_train_full, X_test_new, X_train, y_train, X_test, y_test, rot_mat = \
    im2data.obraz2d2np(fileIm, n, rm, rotate=rotate)

# TabNet требует 2D y
y_train_2d = y_train.reshape(-1, 1)
y_test_2d = y_test.reshape(-1, 1)

print(f"Обучающая выборка: {X_train.shape}")

# Обучение
print("Обучение TabNet...")
start = dt.datetime.now()

model = TabNetRegressor(
    n_d=n_d,
    n_a=n_a,
    n_steps=n_steps,
    gamma=gamma,
    lambda_sparse=lambda_sparse,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
    scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    scheduler_params=dict(T_0=50, eta_min=1e-5),
    mask_type='entmax',  # лучше sparsemax для отбора признаков
    device_name=device,
    verbose=10,
    seed=42,
)

model.fit(
    X_train, y_train_2d,
    eval_set=[(X_test, y_test_2d)],
    eval_metric=['rmse'],
    max_epochs=max_epochs,
    patience=patience,
    batch_size=batch_size,
    virtual_batch_size=virtual_batch_size,
)
train_time = dt.datetime.now() - start

# Предсказания
y_pred_val = model.predict(X_test).ravel()
rmse_val = test_utils.compute_rmse(y_pred_val, y_test)

y_pred_full = test_utils.predict_in_batches(
    lambda x: model.predict(x).ravel(), X_test_new
)
rmse_full = test_utils.compute_rmse(y_pred_full, im.reshape(-1))

test_utils.print_results("TabNet", rmse_full, train_time)
print(f"RMSE на валидации: {rmse_val:.6f}")

# Важность признаков по attention
feat_importances = model.feature_importances_
indices = np.argsort(feat_importances)[::-1][:20]
print("\nТоп-20 признаков по attention-важности:")
for i, idx in enumerate(indices):
    print(f"  {i+1}. Признак {idx}: {feat_importances[idx]:.6f}")


# Визуализация
var = test_utils.show_comparison(
    im, y_pred_full, im.shape,
    method_name="TabNet", rm=rm, rmse=rmse_full
)

test_utils.show_surface_3d(im, "Эталон")
test_utils.show_surface_3d(var, "TabNet")
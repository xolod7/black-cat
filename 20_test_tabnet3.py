# test_tabnet.py
"""
Тест: TabNet — нейросеть с механизмом attention для отбора признаков. (с настройками по умолчанию)
Специально разработана для табличных данных.
pip install pytorch-tabnet
"""
import numpy as np
import torch
import datetime as dt
from pytorch_tabnet.tab_model import TabNetRegressor
import matplotlib.pyplot as plt
import im2data
import test_utils

# ===================== НАСТРОЙКИ =====================
fileIm = test_utils.get_image_path("1.png")
n = 240000
rm = 2             # общее число признаков Раунд 1
# rm = 10            # общее число признаков Раунд 2,3
# rm = 500           # общее число признаков Раунд 4,5     
# rm = 1000          # общее число признаков Раунд 6,7
# rm = 2000          # общее число признаков Раунд 8,9       
rotate = False     # поворот пространства признаков Раунд 1,2,4,6,8
# rotate = True     # поворот пространства признаков Раунд 3,5,7,9

use_gpu = True
device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

# Адаптивные гиперпараметры в зависимости от rm
if rm <= 20:
    max_epochs = 50
elif rm <= 500:
    max_epochs = 150
else:
    max_epochs = 200

results_csv = "results.csv"
# =====================================================

print(f"Device: {device}")
print(f"Загрузка данных: rm={rm}, rotate={rotate}")

im, X_train_full, X_test_new, X_train, y_train, X_test, y_test, rot_mat = \
    im2data.obraz2d2np(fileIm, n, rm, rotate=rotate)

# TabNet требует float32 и 2D y
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
X_train_full = X_train_full.astype(np.float32)
X_test_new = X_test_new.astype(np.float32)
y_train_2d = y_train.reshape(-1, 1).astype(np.float32)
y_test_2d = y_test.reshape(-1, 1).astype(np.float32)

print(f"Обучающая выборка: {X_train.shape}")

# Обучение
print(f"\nОбучение TabNet ({max_epochs} эпох)...")
start = dt.datetime.now()

model = TabNetRegressor(
    device_name=device,
    verbose=10,
    seed=42,
)

model.fit(
    X_train, y_train_2d,
    eval_set=[(X_test, y_test_2d)],
    eval_metric=['rmse'],
    max_epochs=max_epochs,
)
train_time = dt.datetime.now() - start

# Предсказания на валидационной выборке
y_pred_val = model.predict(X_test).ravel()
rmse_val = test_utils.compute_rmse(y_pred_val, y_test)

# Предсказания на тестовой выборке с новым шумом
start_pred = dt.datetime.now()
y_pred_full = test_utils.predict_in_batches(
    lambda x: model.predict(x.astype(np.float32)).ravel(),
    X_test_new,
    batch_size=50000
)
pred_time = dt.datetime.now() - start_pred

rmse_full = test_utils.compute_rmse(y_pred_full, im.reshape(-1))

test_utils.print_results("TabNet", rmse_full, train_time, pred_time)
print(f"RMSE на валидации: {rmse_val:.6f}")
print(f"Лучшая эпоха: {model.best_epoch}")

# Важность признаков по attention
feat_importances = model.feature_importances_
print(f"\nВажность признаков (всего {len(feat_importances)}):")
indices = np.argsort(feat_importances)[::-1]
# Выводим топ — при rm=2 это все признаки
top_n = min(20, len(feat_importances))
for i in range(top_n):
    idx = indices[i]
    print(f"  {i+1}. Признак {idx}: {feat_importances[idx]:.6f}")

# Визуализация
var = test_utils.show_comparison(
    im, y_pred_full, im.shape,
    method_name="TabNet", rm=rm, rmse=rmse_full
)

test_utils.show_surface_3d(im, "Эталон")
test_utils.show_surface_3d(var, "TabNet")
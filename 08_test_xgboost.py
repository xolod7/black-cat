# test_xgboost.py
"""
Тест: XGBoost — один из лучших градиентных бустингов для табличных данных.
Поддерживает GPU через tree_method='gpu_hist'.
"""
import numpy as np
import datetime as dt
import xgboost as xgb
import im2data
import test_utils

# ===================== НАСТРОЙКИ =====================
fileIm = test_utils.get_image_path("1.png")
n = 240000
rm = 2             # общее число признаков Раунд 1
# rm = 10            # общее число признаков Раунд 2,3
# rm = 500           # общее число признаков Раунд 4     
# rm = 1000          # общее число признаков Раунд 6
# rm = 2000          # общее число признаков Раунд 8         
rotate = False     # поворот пространства признаков Раунд 1,2,4,6,8
# rotate = True     # поворот пространства признаков Раунд 3

use_gpu = True  # GPU ускорение

# Гиперпараметры — настроены для задачи с большим количеством
# неинформативных признаков
n_estimators = 100000 

learning_rate = 0.05
max_depth = 6
subsample = 0.8


colsample_bytree = test_utils.adaptive_colsample(rm, target_fraction=0.3, min_features=2)
colsample_bylevel = test_utils.adaptive_colsample(rm, target_fraction=0.5, min_features=2)

# При малом rm ослабляем регуляризацию
if rm <= 20:
    reg_alpha = 0.0
    reg_lambda = 1.0
    min_child_weight = 1
    gamma = 0.0
else:
    reg_alpha = 0.1
    reg_lambda = 1.0
    min_child_weight = 10
    gamma = 0.1
# =====================================================

print(f"Загрузка данных: rm={rm}, rotate={rotate}")
im, X_train_full, X_test_new, X_train, y_train, X_test, y_test, rot_mat = \
    im2data.obraz2d2np(fileIm, n, rm, rotate=rotate)

print(f"Обучающая выборка: {X_train.shape}")

# Обучение
print("Обучение XGBoost...")
if use_gpu:
    tree_method = 'hist'
    device = 'cuda'
else:
    tree_method = 'hist'
    device = 'cpu'

start = dt.datetime.now()
model = xgb.XGBRegressor(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    colsample_bylevel=colsample_bylevel,
    reg_alpha=reg_alpha,
    reg_lambda=reg_lambda,
    min_child_weight=min_child_weight,
    gamma=gamma,
    tree_method=tree_method,
    device=device,
    random_state=42,
    verbosity=1,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100,
)
train_time = dt.datetime.now() - start

# Предсказания
y_pred_val = model.predict(X_test)
rmse_val = test_utils.compute_rmse(y_pred_val, y_test)

y_pred_full = test_utils.predict_in_batches(model.predict, X_test_new)
rmse_full = test_utils.compute_rmse(y_pred_full, im.reshape(-1))

test_utils.print_results("XGBoost", rmse_full, train_time)
print(f"RMSE на валидации: {rmse_val:.6f}")
if hasattr(model, 'best_iteration') and model.best_iteration is not None:
    print(f"Лучшая итерация: {model.best_iteration}")
else:
    print(f"Использованы все {n_estimators} итераций (early stopping не применялся)")

# Важность признаков
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:20]
print("\nТоп-20 признаков по важности:")
for i, idx in enumerate(indices):
    print(f"  {i+1}. Признак {idx}: {importances[idx]:.6f}")


# Визуализация
var = test_utils.show_comparison(
    im, y_pred_full, im.shape,
    method_name="XGBoost", rm=rm, rmse=rmse_full
)

test_utils.show_surface_3d(im, "Эталон")
test_utils.show_surface_3d(var, "XGBoost")
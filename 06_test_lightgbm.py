# test_lightgbm.py
"""
Тест: LightGBM — быстрый градиентный бустинг, особенно эффективен
при большом количестве признаков благодаря алгоритму GOSS и
поддержке feature bundling.
Поддерживает GPU.
"""
import numpy as np
import datetime as dt
import lightgbm as lgb
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

# Гиперпараметры
n_estimators = 5000     # Раунд 1
# n_estimators = 20000     # Раунд 2,3
# n_estimators = 30000     # Раунд 4,6,8

use_gpu = True

learning_rate = 0.05
num_leaves = 63         # основной параметр сложности
max_depth = -1           # без ограничения
subsample = 0.8

colsample_bytree = test_utils.adaptive_colsample(rm, target_fraction=0.3, min_features=2)

if rm <= 20:
    reg_alpha = 0.0
    reg_lambda = 0.0
    min_child_samples = 5
else:
    reg_alpha = 0.1
    reg_lambda = 1.0
    min_child_samples = 20

early_stopping_rounds = 1000
# =====================================================

print(f"Загрузка данных: rm={rm}, rotate={rotate}")
im, X_train_full, X_test_new, X_train, y_train, X_test, y_test, rot_mat = \
    im2data.obraz2d2np(fileIm, n, rm, rotate=rotate)

print(f"Обучающая выборка: {X_train.shape}")

# Обучение
print("Обучение LightGBM...")

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': n_estimators,
    'learning_rate': learning_rate,
    'num_leaves': num_leaves,
    'max_depth': max_depth,
    'subsample': subsample,
    'colsample_bytree': colsample_bytree,
    'reg_alpha': reg_alpha,
    'reg_lambda': reg_lambda,
    'min_child_samples': min_child_samples,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}

if use_gpu:
    params['device'] = 'gpu'
    params['gpu_use_dp'] = False  # float32 на GPU

start = dt.datetime.now()
model = lgb.LGBMRegressor(**params)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[
        lgb.log_evaluation(100),
        lgb.early_stopping(early_stopping_rounds)
    ]
)
train_time = dt.datetime.now() - start

# Предсказания
y_pred_val = model.predict(X_test)
rmse_val = test_utils.compute_rmse(y_pred_val, y_test)

y_pred_full = test_utils.predict_in_batches(model.predict, X_test_new)
rmse_full = test_utils.compute_rmse(y_pred_full, im.reshape(-1))

test_utils.print_results("LightGBM", rmse_full, train_time)
print(f"RMSE на валидации: {rmse_val:.6f}")
print(f"Лучшая итерация: {model.best_iteration_}")

# Важность признаков
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:20]
print("\nТоп-20 признаков по важности:")
for i, idx in enumerate(indices):
    print(f"  {i+1}. Признак {idx}: {importances[idx]:.4f}")


# Визуализация
var = test_utils.show_comparison(
    im, y_pred_full, im.shape,
    method_name="LightGBM", rm=rm, rmse=rmse_full
)

test_utils.show_surface_3d(im, "Эталон")
test_utils.show_surface_3d(var, "LightGBM")
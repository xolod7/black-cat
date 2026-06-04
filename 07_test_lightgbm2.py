# test_lightgbm.py
"""
Тест: LightGBM — быстрый градиентный бустинг (настройки по умолчанию), особенно эффективен
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

use_gpu = True

# Гиперпараметры
n_estimators = 30000

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
    'random_state': 42,
    'n_jobs': -1,
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
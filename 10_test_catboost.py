# test_catboost.py
"""
Тест: CatBoost — градиентный бустинг от Яндекса.
Хорошо работает «из коробки», имеет встроенную поддержку GPU.
Использует ordered boosting для борьбы с переобучением.
"""
import numpy as np
import datetime as dt
from catboost import CatBoostRegressor
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
iterations = 50000

learning_rate = 0.05
depth = 8
l2_leaf_reg = 3.0
subsample = 0.8
colsample_bylevel = 0.3   # аналог colsample_bytree
random_strength = 1.0
early_stopping_rounds = 1000


print(f"Загрузка данных: rm={rm}, rotate={rotate}")
im, X_train_full, X_test_new, X_train, y_train, X_test, y_test, rot_mat = \
    im2data.obraz2d2np(fileIm, n, rm, rotate=rotate)

print(f"Обучающая выборка: {X_train.shape}")

# Обучение
print("Обучение CatBoost...")
start = dt.datetime.now()

params = dict(
    iterations=iterations,
    learning_rate=learning_rate,
    depth=depth,
    l2_leaf_reg=l2_leaf_reg,
    bootstrap_type='Bernoulli',
    subsample=subsample,
    random_strength=random_strength,
    task_type='GPU' if use_gpu else 'CPU',
    devices='0',
    random_seed=42,
    verbose=100,
    early_stopping_rounds=early_stopping_rounds,
    loss_function='RMSE',
)


# RSM не поддерживается на GPU; на CPU адаптируем
if not use_gpu:
    params['colsample_bylevel'] = test_utils.adaptive_colsample(
        rm, target_fraction=0.3, min_features=2
    )

model = CatBoostRegressor(**params)

model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    use_best_model=True
)
train_time = dt.datetime.now() - start

# Предсказания
y_pred_val = model.predict(X_test)
rmse_val = test_utils.compute_rmse(y_pred_val, y_test)

y_pred_full = test_utils.predict_in_batches(model.predict, X_test_new)
rmse_full = test_utils.compute_rmse(y_pred_full, im.reshape(-1))

test_utils.print_results("CatBoost", rmse_full, train_time)
print(f"RMSE на валидации: {rmse_val:.6f}")
print(f"Лучшая итерация: {model.best_iteration_}")

# Важность признаков
importances = model.get_feature_importance()
indices = np.argsort(importances)[::-1][:20]
print("\nТоп-20 признаков по важности:")
for i, idx in enumerate(indices):
    print(f"  {i+1}. Признак {idx}: {importances[idx]:.4f}")

# Визуализация
var = test_utils.show_comparison(
    im, y_pred_full, im.shape,
    method_name="CatBoost", rm=rm, rmse=rmse_full
)

test_utils.show_surface_3d(im, "Эталон")
test_utils.show_surface_3d(var, "CatBoost")


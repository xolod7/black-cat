# test_knn.py
"""
Тест: метод k ближайших соседей (KNeighborsRegressor).
Ожидание: плохо работает при большом числе признаков (проклятие размерности).
"""
import numpy as np
import datetime as dt
from sklearn.neighbors import KNeighborsRegressor
import im2data
import test_utils

# ===================== НАСТРОЙКИ =====================
fileIm = test_utils.get_image_path("1.png")
n = 240000
rm = 2               # общее число признаков Раунд 1
# rm = 10              # общее число признаков Раунд 2
rotate = False       # поворот пространства признаков

n_neighbors = 5      # количество соседей
# При большом rm используем алгоритм ball_tree или kd_tree
# ball_tree лучше работает в высокоразмерных пространствах
algorithm = 'auto'
n_jobs = -1          # все ядра CPU
# =====================================================

print(f"Загрузка данных: rm={rm}, rotate={rotate}")
im, X_train_full, X_test_new, X_train, y_train, X_test, y_test, rot_mat = \
    im2data.obraz2d2np(fileIm, n, rm, rotate=rotate)

print(f"Обучающая выборка: {X_train.shape}")
print(f"Тестовая выборка: {X_test.shape}")

# Обучение
print("Обучение KNN...")
start = dt.datetime.now()
model = KNeighborsRegressor(
    n_neighbors=n_neighbors,
    algorithm=algorithm,
    weights='distance',  # взвешивание по расстоянию — чуть лучше
    n_jobs=n_jobs
)
model.fit(X_train, y_train)
train_time = dt.datetime.now() - start

# Предсказание на валидационной выборке
start = dt.datetime.now()
y_pred_val = model.predict(X_test)
pred_time_val = dt.datetime.now() - start
rmse_val = test_utils.compute_rmse(y_pred_val, y_test)

# Предсказание на полной тестовой выборке (с новым шумом)
start = dt.datetime.now()
y_pred_full = test_utils.predict_in_batches(model.predict, X_test_new, batch_size=50000)
pred_time_full = dt.datetime.now() - start
rmse_full = test_utils.compute_rmse(y_pred_full, im.reshape(-1))

test_utils.print_results(
    f"KNN (k={n_neighbors})", rmse_full, train_time, pred_time_full
)
print(f"RMSE на валидации: {rmse_val:.6f}")

# Визуализация
var = test_utils.show_comparison(
    im, y_pred_full, im.shape,
    method_name="KNN", rm=rm, rmse=rmse_full
)

test_utils.show_surface_3d(im, "Эталон")
test_utils.show_surface_3d(var, f"KNN k={n_neighbors}")
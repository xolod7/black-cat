# test_ridge.py
"""
Тест: Ridge регрессия (линейная регрессия с L2-регуляризацией).
Ожидание: не может выучить сложную нелинейную зависимость,
но будет интересно увидеть нижнюю границу — что даёт линейный метод.
"""
import numpy as np
import datetime as dt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import im2data
import test_utils

# ===================== НАСТРОЙКИ =====================
fileIm = test_utils.get_image_path("1.png")
n = 240000
rm = 2          # общее число признаков
rotate = False  # поворот пространства признаков
alpha = 1.0  # коэффициент регуляризации
# =====================================================

print(f"Загрузка данных: rm={rm}, rotate={rotate}")
im, X_train_full, X_test_new, X_train, y_train, X_test, y_test, rot_mat = \
    im2data.obraz2d2np(fileIm, n, rm, rotate=rotate)

print(f"Обучающая выборка: {X_train.shape}")

# Обучение
print("Обучение Ridge регрессии...")
start = dt.datetime.now()
model = Ridge(alpha=alpha)
model.fit(X_train, y_train)
train_time = dt.datetime.now() - start

# Предсказания
y_pred_val = model.predict(X_test)
rmse_val = test_utils.compute_rmse(y_pred_val, y_test)

y_pred_full = test_utils.predict_in_batches(model.predict, X_test_new)
rmse_full = test_utils.compute_rmse(y_pred_full, im.reshape(-1))

test_utils.print_results("Ridge Regression", rmse_full, train_time)
print(f"RMSE на валидации: {rmse_val:.6f}")

# Визуализация
var = test_utils.show_comparison(
    im, y_pred_full, im.shape,
    method_name="Ridge", rm=rm, rmse=rmse_full
)

test_utils.show_surface_3d(im, "Эталон")
test_utils.show_surface_3d(var, "Ridge")
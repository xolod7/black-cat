# test_lightgbm_pca.py
"""
Тест: LightGBM + PCA (Метод Главных Компонент).
Проверка гипотезы: спасет ли выравнивание осей деревья от матричного поворота?
PCA обучается ТОЛЬКО на обучающей выборке, затем применяется к валидации и тесту.
"""
import numpy as np
import datetime as dt
import lightgbm as lgb
from sklearn.decomposition import PCA
import im2data
import test_utils

# ===================== НАСТРОЙКИ =====================
fileIm = test_utils.get_image_path("1.png")
n = 240000
rm = 2000
rotate = True # Самый сложный режим

# Настройки PCA
pca_components = 50 # Сжимаем 2000 признаков до 50 главных компонент
# =====================

print(f"Загрузка данных: rm={rm}, rotate={rotate}")
im, X_train_full, X_test_new, X_train, y_train, X_test, y_test, rot_mat = \
im2data.obraz2d2np(fileIm, n, rm, rotate=rotate)

print(f"Обучающая выборка ДО PCA: {X_train.shape}")

# ----------------- PCA ТРАНСФОРМАЦИЯ -----------------
print(f"\nПрименение PCA (сжатие до {pca_components} компонент)...")
start_pca = dt.datetime.now()

# Обучаем PCA только на train!
pca = PCA(n_components=pca_components)
X_train_pca = pca.fit_transform(X_train)

# Применяем обученную PCA к валидации и финальному тесту
X_test_pca = pca.transform(X_test)
X_test_new_pca = pca.transform(X_test_new)

pca_time = dt.datetime.now() - start_pca
print(f"PCA выполнена за {pca_time}.")
print(f"Сохранено объясненной дисперсии: {sum(pca.explained_variance_ratio_):.4f} ({pca_components} компонент)")
print(f"Размерность после PCA: {X_train_pca.shape}")

# ----------------- OБУЧЕНИЕ LIGHTGBM -----------------
print("\nОбучение LightGBM на PCA-данных...")
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'n_estimators': 30000,
    'random_state': 42,
    'n_jobs': -1,
    'device': 'gpu', # Используем GPU
    'gpu_use_dp': False 
}

start = dt.datetime.now()
model = lgb.LGBMRegressor(**params)

model.fit(
    X_train_pca, y_train,
    eval_set=[(X_test_pca, y_test)],
    callbacks=[
        lgb.log_evaluation(100),
        lgb.early_stopping(1000)
    ]
)
train_time = dt.datetime.now() - start

# ----------------- ОЦЕНКА -----------------
y_pred_val = model.predict(X_test_pca)
rmse_val = test_utils.compute_rmse(y_pred_val, y_test)

y_pred_full = test_utils.predict_in_batches(model.predict, X_test_new_pca, batch_size=50000)
rmse_full = test_utils.compute_rmse(y_pred_full, im.reshape(-1))

# Добавляем время PCA к общему времени обучения
total_time = train_time + pca_time

test_utils.print_results(f"LightGBM + PCA({pca_components})", rmse_full, total_time)
print(f"RMSE на валидации: {rmse_val:.6f}")
print(f"Лучшая итерация: {model.best_iteration_}")

# ----------------- ВИЗУАЛИЗАЦИЯ -----------------
var = test_utils.show_comparison(
    im, y_pred_full, im.shape,
    method_name=f"LightGBM + PCA({pca_components})", rm=rm, rmse=rmse_full
)

test_utils.show_surface_3d(im, "Эталон")
test_utils.show_surface_3d(var, f"LightGBM + PCA({pca_components})")
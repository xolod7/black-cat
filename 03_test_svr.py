# test_svr.py
"""
Тест: Support Vector Regression с RBF ядром.
ВНИМАНИЕ: SVR имеет сложность O(n²)–O(n³), поэтому при 240000 примерах
работает катастрофически долго. Используем подвыборку.
Но метод интересен как пример kernel-метода.
"""
import numpy as np
import datetime as dt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import im2data
import test_utils

# ===================== НАСТРОЙКИ =====================
fileIm = test_utils.get_image_path("1.png")
n = 240000
rm = 2   # общее число признаков, SVR не потянет 1000 признаков на 240k примерах
rotate = False  # поворот пространства признаков

# Подвыборка для SVR
svr_train_size = 10000  # используем только 10000 примеров!

C = 10.0
epsilon = 0.01
gamma = 'scale'
# =====================================================

print(f"Загрузка данных: rm={rm}, rotate={rotate}")
print(f"ВНИМАНИЕ: SVR использует подвыборку {svr_train_size} из {n} примеров!")

im, X_train_full, X_test_new, X_train, y_train, X_test, y_test, rot_mat = \
    im2data.obraz2d2np(fileIm, n, rm, rotate=rotate)

# Подвыборка
idx = np.random.RandomState(42).permutation(len(X_train))[:svr_train_size]
X_train_sub = X_train[idx]
y_train_sub = y_train[idx]

print(f"Обучающая подвыборка: {X_train_sub.shape}")

# Обучение
print("Обучение SVR...")
start = dt.datetime.now()
model = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma))
])
model.fit(X_train_sub, y_train_sub)
train_time = dt.datetime.now() - start

# Предсказания
y_pred_val = model.predict(X_test)
rmse_val = test_utils.compute_rmse(y_pred_val, y_test)

y_pred_full = test_utils.predict_in_batches(model.predict, X_test_new)
rmse_full = test_utils.compute_rmse(y_pred_full, im.reshape(-1))

test_utils.print_results(
    f"SVR (train={svr_train_size})", rmse_full, train_time
)
print(f"RMSE на валидации: {rmse_val:.6f}")

# Визуализация
var = test_utils.show_comparison(
    im, y_pred_full, im.shape,
    method_name="SVR", rm=rm, rmse=rmse_full
)

test_utils.show_surface_3d(im, "Эталон")
test_utils.show_surface_3d(var, "SVR")
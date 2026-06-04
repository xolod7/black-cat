"""
Тест: полигармонический каскад.
"""
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import sys
import datetime as dt
import os
import im2data
import test_utils

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import collective

# ===================== НАСТРОЙКИ =====================
fileIm = test_utils.get_image_path("1.png")
n = 240000

# Раунд 1
rm = 2              # общее число признаков
epoh = 10           # общее число признаков
# epoh = 20           # общее число признаков
# epoh = 50           # общее число признаков
# epoh = 100          # общее число признаков
rotate = False      # поворот пространства признаков

# Раунд 2,3
# rm = 10             # общее число признаков
# epoh = 10           # общее число признаков
# epoh = 20           # общее число признаков
# epoh = 50           # общее число признаков
# epoh = 100          # общее число признаков
# rotate = False      # поворот пространства признаков Раунд 2
# rotate = True       # поворот пространства признаков Раунд 3

# Раунд 4,5
# rm = 500            # общее число признаков
# epoh = 30           # общее число признаков
# epoh = 50           # общее число признаков
# epoh = 150          # общее число признаков
# epoh = 300          # общее число признаков
# rotate = False      # поворот пространства признаков Раунд 4
# rotate = True       # поворот пространства признаков Раунд 5

# Раунд 6,7,8,9
# rm = 1000           # общее число признаков Раунд 6,7
# rm = 2000           # общее число признаков Раунд 8,9
# epoh = 50           # общее число признаков
# epoh = 100          # общее число признаков
# epoh = 200          # общее число признаков
# epoh = 500          # общее число признаков
# rotate = False      # поворот пространства признаков Раунд 6,8
# rotate = True       # поворот пространства признаков Раунд 7,9


type_calc = "float"
mode = "gpu"       # "gpu" или "cpu"
batch = 3000
func = "reg"
mollis = 4

# Схема каскада: 12 слоёв
schema = [rm, 3, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 1]
# =====================================================

print(f"Загрузка данных: rm={rm}, rotate={rotate}")
im, X_train_full, X_test_new, X_train, y_train, X_test, y_test, rot_mat = \
    im2data.obraz2d2(fileIm, n, rm, rotate=rotate)

print(f"Обучающая выборка: {X_train.shape}")

# Обучение
print("Обучение полигармонического каскада...")
start = dt.datetime.now()

pc = collective.Collective(schema, type_calc, mode, func)
erl, erv = pc.Disciplina(X_train, y_train, X_test, y_test, batch, epoh, mollis)

train_time = dt.datetime.now() - start

# График обучения
test_utils.plot_learning_curve(erl, erv, "Полигармонический каскад: кривая обучения")

# Предсказания на полной тестовой выборке
start = dt.datetime.now()
var = pc.FlammaPars(X_test_new, 50000)
pred_time = dt.datetime.now() - start

# var[var < 0] = 0
# var[var > 1] = 1
var = var.reshape(im.shape[0], im.shape[1])

# Ошибка
rmse_full = torch.mean((var - im) ** 2) ** 0.5
rmse_full = rmse_full.item()

test_utils.print_results("Полигармонический каскад", rmse_full, train_time, pred_time)

# Визуализация
var = test_utils.show_comparison(
    im, var.numpy(), im.shape,
    method_name="Полигармонический каскад", rm=rm, rmse=rmse_full
)

test_utils.show_surface_3d(im, "Эталон")
test_utils.show_surface_3d(var, "Полигармонический каскад")

# EL PSY KONGROO
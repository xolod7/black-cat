# # test_utils.py
"""
Общие утилиты для всех тестов: визуализация, замер времени, вычисление метрик.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import datetime as dt
import os

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def compute_rmse(predictions, targets):
    """Вычисление RMSE."""
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions).float()
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets).float()
    predictions = predictions.reshape(-1)
    targets = targets.reshape(-1)
    return torch.sqrt(torch.mean((predictions - targets) ** 2)).item()


def clip_predictions(var):
    """Обрезка предсказаний в диапазон [0, 1]."""
    if isinstance(var, np.ndarray):
        var = np.clip(var, 0, 1)
    else:
        var = torch.clamp(var, 0, 1)
    return var


def show_image_from_predictions(predictions, shape, title="Результат"):
    """Показать изображение из предсказаний."""
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions).float()
    var = clip_predictions(predictions)
    var = var.reshape(shape[0], shape[1])
    convert_image = transforms.ToPILImage()
    image = convert_image(var)
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
    return var


def show_original_image(im_tensor):
    """Показать исходное изображение."""
    convert_image = transforms.ToPILImage()
    image = convert_image(im_tensor)
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title("Эталон")
    plt.axis('off')
    plt.show()


def show_surface_3d(z_tensor, title="Поверхность"):
    """Показать 3D-поверхность (требуется plotly)."""
    if not HAS_PLOTLY:
        print("plotly не установлен, 3D-визуализация пропущена")
        return
    if isinstance(z_tensor, torch.Tensor):
        z_data = z_tensor.numpy()
    else:
        z_data = z_tensor
    fig = go.Figure(data=[go.Surface(z=z_data, colorscale='Gray')])
    fig.update_scenes(camera_projection_type='orthographic')
    fig.update_layout(title=title)
    fig.show()


def plot_learning_curve(train_errors, val_errors=None, title="Кривая обучения"):
    """График кривой обучения."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_errors, label='train', linestyle='--', alpha=0.8)
    if val_errors is not None:
        plt.plot(val_errors, label='val', linewidth=2)
    plt.grid(True)
    plt.xlabel('эпохи')
    plt.ylabel('RMSE')
    plt.title(title)
    plt.legend()
    plt.show()


def print_results(method_name, rmse, train_time, test_time=None):
    """Печать результатов."""
    print(f"\n{'='*50}")
    print(f"Метод: {method_name}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Время обучения: {train_time}")
    if test_time is not None:
        print(f"Время предсказания: {test_time}")
    print(f"{'='*50}\n")


def predict_in_batches(model_predict_func, X, batch_size=50000):
    """Предсказание батчами для экономии памяти."""
    n = X.shape[0]
    results = []
    for i in range(0, n, batch_size):
        batch = X[i:i + batch_size]
        pred = model_predict_func(batch)
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        results.append(pred.ravel())
    return np.concatenate(results)


# Путь к файлу изображения
def get_image_path(filename="1.png"):
    path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(path, filename)


def show_comparison(im_original, predictions, shape, method_name="", rm=0, rmse=None):
    """
    Показать эталон и результат бок о бок в одном окне.
    
    Параметры:
    ----------
    im_original : torch.Tensor
        Эталонное изображение.
    predictions : numpy.ndarray или torch.Tensor
        Предсказания модели (одномерный массив).
    shape : tuple
        Форма изображения (H, W).
    method_name : str
        Название метода для заголовка.
    rm : int
        Количество признаков для заголовка.
    rmse : float или None
        RMSE для вывода в заголовке.
    
    Возвращает:
    ----------
    var : numpy.ndarray
        Обрезанные предсказания в форме изображения (H, W).
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()
    if isinstance(im_original, torch.Tensor):
        im_np = im_original.numpy()
    else:
        im_np = im_original
    
    var = clip_predictions(predictions.copy())
    var = var.reshape(shape[0], shape[1])
    
    # Вычисляем разницу для третьего subplot
    diff = np.abs(im_np - var)
    
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    
    axes[0].imshow(im_np, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Эталон", fontsize=14)
    axes[0].axis('off')
    
    title_result = method_name
    if rm > 0:
        title_result += f", rm={rm}"
    if rmse is not None:
        title_result += f"\nRMSE={rmse:.6f}"
    axes[1].imshow(var, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(title_result, fontsize=14)
    axes[1].axis('off')
    
    im_diff = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
    axes[2].set_title("Карта ошибки |эталон − результат|", fontsize=14)
    axes[2].axis('off')
    plt.colorbar(im_diff, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
    
    return var


def adaptive_colsample(rm, target_fraction=0.3, min_features=2):
    """
    Вычисляет долю признаков для случайного отбора,
    гарантируя что будет выбрано не менее min_features.
    
    При rm=2:    возвращает 1.0 (все признаки)
    При rm=10:   возвращает 0.5 (минимум 2 из 10... но 0.3*10=3, можно 0.3)
    При rm=1000: возвращает 0.3
    """
    min_fraction = min_features / rm
    return max(target_fraction, min_fraction)


def adaptive_max_features_tree(rm, strategy='sqrt', min_features=2):
    """
    Для древесных методов sklearn (Random Forest, ExtraTrees).
    
    При rm=2:    возвращает 1.0 (оба признака)
    При rm=10:   возвращает max(sqrt(10), 2)/10 ≈ 0.32
    При rm=1000: возвращает 'sqrt'
    """
    if rm <= min_features * 2:
        return 1.0
    if strategy == 'sqrt':
        import math
        needed = max(math.sqrt(rm), min_features)
        if needed / rm > 0.5:
            return 1.0
        return strategy
    return strategy
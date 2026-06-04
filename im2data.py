import torch
from torchvision import transforms
from PIL import Image
import numpy as np


def _random_rotation_matrix(dim, seed=42):
    """
    Генерирует случайную ортогональную матрицу размера dim x dim
    используя QR-разложение случайной матрицы.
    Это эквивалентно случайному повороту в dim-мерном пространстве.
    """
    rng = np.random.RandomState(seed)
    H = rng.randn(dim, dim)
    Q, R = np.linalg.qr(H)
    # Обеспечиваем детерминант +1 (собственно поворот, а не отражение)
    d = np.diag(R)
    ph = np.sign(d)
    Q = Q * ph[np.newaxis, :]
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def _generate_data(file, n, rm, rotate=False, rotation_seed=42):
    """
    Внутренняя функция генерации данных.
    
    Параметры:
    ----------
    file : str
        Путь к файлу изображения.
    n : int
        Размер обучающей выборки.
    rm : int
        Общее количество входных признаков (должно быть чётным, >= 2).
        Из них 2 информативных, остальные — шум.
    rotate : bool
        Если True, применяется случайный поворот в пространстве признаков.
    rotation_seed : int
        Seed для генерации матрицы поворота (для воспроизводимости).
    
    Возвращает:
    ----------
    mas : torch.Tensor
        Исходное изображение как тензор (ny, nx), значения [0,1].
    mtk : torch.Tensor
        Полный набор признаков (kol, rm) — упорядочен как пиксели изображения.
        Для восстановления изображения из предсказаний обученной модели.
    mtk2 : torch.Tensor
        Полный набор признаков (kol, rm) с ДРУГИМИ случайными шумовыми
        значениями, но теми же информативными x,y. 
        Для проверки на переобученность.
    tk : torch.Tensor
        Обучающая выборка, входы (n, rm).
    rez : torch.Tensor
        Обучающая выборка, выходы (n, 1).
    tx : torch.Tensor
        Валидационная выборка, входы (kol-n, rm).
    rz : torch.Tensor
        Валидационная выборка, выходы (kol-n, 1).
    rotation_matrix : numpy.ndarray или None
        Матрица поворота (rm, rm), если rotate=True, иначе None.
    """
    rm_half = round(rm / 2)
    
    image = Image.open(file)
    convert_tensor = transforms.ToTensor()
    mas = convert_tensor(image)
    mas = torch.mean(mas, dim=0)
    mn = torch.min(mas)
    mx = torch.max(mas)
    mas = (mas - mn) / (mx - mn)

    torch.manual_seed(10)
    nx = mas.shape[0]
    ny = mas.shape[1]
    kol = nx * ny

    x = torch.tensor(range(nx))
    y = torch.tensor(range(ny))

    X, Y = torch.meshgrid(x, y, indexing='ij')
    X = X.float() / nx - 0.5
    Y = Y.float() / ny - 0.5
    X = X.reshape(kol, 1)
    Y = Y.reshape(kol, 1)
    mt = torch.cat([X, Y], dim=1)
    mrez = mas.reshape(kol, 1)

    # Создание набора признаков с шумом (набор 1 — для обучения)
    mtk = mt.clone()
    mtk = mtk.repeat(1, rm_half)
    # Создание набора признаков с шумом (набор 2 — для теста на переобучение)
    mtk2 = mt.clone()
    mtk2 = mtk2.repeat(1, rm_half)
    
    for i in range(1, rm_half):
        n1 = 2 * i
        n2 = n1 + 2
        id = torch.randperm(kol)
        mtk[:, n1:n2] = mt[id, :]
        id = torch.randperm(kol)
        mtk2[:, n1:n2] = mt[id, :]
    
    # Перемешиваем порядок столбцов (чтобы информативные не были первыми)
    id = torch.randperm(2 * rm_half)
    mtk = mtk[:, id]
    mtk2 = mtk2[:, id]

    # Применяем поворот в пространстве признаков
    rotation_matrix = None
    if rotate and rm > 2:
        dim = mtk.shape[1]
        rotation_matrix = _random_rotation_matrix(dim, seed=rotation_seed)
        rot_tensor = torch.from_numpy(rotation_matrix).float()
        mtk = mtk @ rot_tensor.T
        mtk2 = mtk2 @ rot_tensor.T

    # Перемешиваем строки для разделения на обучающую и валидационную
    id = torch.randperm(kol)
    mtk3 = mtk[id, :]
    mrez_shuffled = mrez[id, :]

    tk = mtk3[:n, :]
    rez = mrez_shuffled[:n, :]
    tx = mtk3[n:, :]
    rz = mrez_shuffled[n:, :]

    return mas, mtk, mtk2, tk, rez, tx, rz, rotation_matrix


def obraz2d2(file, n, rm, rotate=False, rotation_seed=42):
    """
    Генерация выборок в формате torch.Tensor.
    
    Параметры и возвращаемые значения — см. _generate_data.
    Возвращает тензоры PyTorch.
    """
    mas, mtk, mtk2, tk, rez, tx, rz, rot_mat = _generate_data(
        file, n, rm, rotate, rotation_seed
    )
    return mas, mtk, mtk2, tk, rez, tx, rz, rot_mat


def obraz2d2np(file, n, rm, rotate=False, rotation_seed=42):
    """
    Генерация выборок в формате numpy.ndarray.
    
    Параметры и возвращаемые значения — см. _generate_data.
    Возвращает numpy-массивы (кроме mas, который остаётся тензором для визуализации).
    """
    mas, mtk, mtk2, tk, rez, tx, rz, rot_mat = _generate_data(
        file, n, rm, rotate, rotation_seed
    )
    return (mas, mtk.numpy(), mtk2.numpy(), tk.numpy(), rez.numpy().ravel(),
            tx.numpy(), rz.numpy().ravel(), rot_mat)
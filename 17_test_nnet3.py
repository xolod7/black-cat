from torchvision import transforms
import matplotlib.pyplot as plt
import datetime as dt
# plotly - для отображения 3d поверхности (можно отключить)
import plotly.graph_objects as go
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import math
import os
import im2data
import test_utils

fileIm = test_utils.get_image_path("1.png")
n = 240000
rm = 2             # общее число признаков Раунд 1
# rm = 10            # общее число признаков Раунд 2,3
# rm = 500           # общее число признаков Раунд 4,5     
# rm = 1000          # общее число признаков Раунд 6,7
# rm = 2000          # общее число признаков Раунд 8,9       
rotate = False     # поворот пространства признаков Раунд 1,2,4,6,8
# rotate = True     # поворот пространства признаков Раунд 3,5,7,9

EPOCHS = 200       # раунд 1,2,3
# EPOCHS = 400       # раунд 4,5
# EPOCHS = 1000       # раунд 6,7
# EPOCHS = 2000       # раунд 8,9

use_gpu = False    # почему-то на моём железе, на cpu эта нейросеть работала быстрее
device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')


print(f"Device: {device}")
print(f"Загрузка данных: rm={rm}, rotate={rotate}")

im, X_train_full, X_test_new, X_train, y_train, X_test, y_test, rot_mat = \
    im2data.obraz2d2(fileIm, n, rm, rotate=rotate)


class MyDataset(Dataset):
  
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]
  
    def __getitem__(self, index):
        return (self.X[index], self.y[index])
    
train = MyDataset(X_train,y_train)
test = MyDataset(X_test, y_test)
trainset = DataLoader(train, batch_size=500, shuffle=True)
testset = DataLoader(test, batch_size=30000, shuffle=False)

ReLU = nn.ReLU()

# Нейросеть
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(rm, 3).to(device)
        self.fc2 = nn.Linear(3, 50).to(device)
        self.fc3 = nn.Linear(50, 50).to(device)
        self.fc4 = nn.Linear(50, 50).to(device)
        self.fc5 = nn.Linear(50, 50).to(device)
        self.fc6 = nn.Linear(50, 50).to(device)
        self.fc7 = nn.Linear(50, 50).to(device)
        self.fc8 = nn.Linear(50, 50).to(device)
        self.fc9 = nn.Linear(50, 50).to(device)
        self.fc10 = nn.Linear(50, 1).to(device)

    def forward(self, x):
        x = ReLU(self.fc1(x))
        x = ReLU(self.fc2(x))
        x = ReLU(self.fc3(x))
        x = ReLU(self.fc4(x))
        x = ReLU(self.fc5(x))
        x = ReLU(self.fc6(x))
        x = ReLU(self.fc7(x))
        x = ReLU(self.fc8(x))
        x = ReLU(self.fc9(x))
        x = self.fc10(x)
        return x
    
# Создание нейросети
# torch.manual_seed(0)
startL = dt.datetime.now()
model = Net()

# Целевая функция
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
# criterion = nn.NLLLoss()
# Оптимизатор
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)

# Обучение
train_loss = []
val_loss = []
for epoch in range(EPOCHS):
    start = dt.datetime.now()
    print(f'Эпоха: {epoch}', end=' ')
    mean_loss = 0
    batch_n = 0
    for X, y in trainset:
        model.zero_grad()
        output = model(X.to(device))

        loss = criterion(output, y.to(device))
        loss.backward()
        optimizer.step()

        mean_loss += loss
        batch_n += 1

    mean_loss /= batch_n
    mean_loss = math.sqrt(mean_loss)
    train_loss.append(mean_loss)
    print(f'Ошибка обучения, RMSE: {mean_loss}, {dt.datetime.now() - start} сек')

    mean_loss = 0
    batch_n = 0
    with torch.no_grad():
        for X, y in testset:
            output = model(X.to(device))
            loss = criterion(output, y.to(device))

            mean_loss += loss
            batch_n += 1
    
    mean_loss /= batch_n
    mean_loss = math.sqrt(mean_loss)
    val_loss.append(mean_loss)
    print(f'Ошибка валидации, RMSE: {mean_loss}')

train_time = dt.datetime.now() - startL

# Построение графика обучения
def plot_acc(Loss_train, Loss_val):
    plt.plot(range(len(Loss_train)), Loss_train, color='orange', label='train', linestyle='--')
    plt.plot(range(len(Loss_val)), Loss_val, color='blue', label='val')
    plt.grid(True)
    plt.xlabel('эпохи')
    plt.ylabel('Ошибка, RMSE')    
    plt.legend()
    plt.show()

plot_acc(train_loss, val_loss)

# Проверка результата

# Обработка данных использованных при обучении для создания изображения
start = dt.datetime.now()
var = model(X_test_new.to(device))
pred_time = dt.datetime.now() - start
var[var<0] = 0
var[var>1] = 1
var = var.reshape(im.shape[0],im.shape[1])
var = var.to('cpu')

# Ошибка
rmse_full = torch.mean((var - im) ** 2) ** 0.5
rmse_full = rmse_full.item()

test_utils.print_results("Neural Network", rmse_full, train_time, pred_time)

# Визуализация
var = test_utils.show_comparison(
    im, var.detach().numpy(), im.shape,
    method_name="Neural Network", rm=rm, rmse=rmse_full
)

test_utils.show_surface_3d(im, "Эталон")
test_utils.show_surface_3d(var, "Neural Network")
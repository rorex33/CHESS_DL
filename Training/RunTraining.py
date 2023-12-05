import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import gym
import gym_chess
import os
import chess
from tqdm import tqdm
from gym_chess.alphazero.move_encoding import utils
from pathlib import Path
from typing import Optional

from Model import *
from MainTraining import *
from TrainingEncodDecod import *


# Гиперпараметры
EPOCHS = 500
LEARNING_RATE = 0.001
MOMENTUM = 0.9

# Запустить тренировку

createBestModelFile()

bestLoss, bestModelPath = retrieveBestModelInfo()

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('Training/runs/fashion_trainer_{}'.format(timestamp))
epoch_number = 0

model = Model()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

best_vloss = 1_000_000.

for epoch in tqdm(range(EPOCHS)):
    if (epoch_number % 5 == 0):
        print('EPOCH {}:'.format(epoch_number + 1))

    # Убедиться, что градиент-трекинг включен и сделать проход по данным
    model.train(True)
    avg_loss = train_one_epoch(model, optimizer, loss_fn, epoch_number, writer)

    running_vloss = 0.0
    # Установить модель в режим оценки, отключив отсев и используя совокупность.
    # статистика для пакетной нормализации.

    model.eval()

    # Отключите вычисление градиента и уменьшите потребление памяти.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)

            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)

    # Писать в консоль только раз в 5 эпох
    if epoch_number % 5 == 0:
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Логировать средние текущие потери на партию
    # как для обучения, так и для валидации
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Отследить лучший результат и сохранить состояние модели
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss

        if (bestLoss > best_vloss): # Если потери лучше, чем у всех предыдущих моделей, то сохранить её
            model_path = 'Training/savedModels/model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)
            saveBestModel(best_vloss, model_path)

    epoch_number += 1

print("\n\nBEST VALIDATION LOSS FOR ALL MODELS: ", bestLoss)
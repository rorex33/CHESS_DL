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


# ПАРСИНГ КОНФИГА #
config = configparser.ConfigParser()
config.readfp(open(r'../config.txt'))

epochs = int(config.get('RunTraining', 'EPOCHS'))
learningRate = float(config.get('RunTraining', 'LEARNING_RATE'))
momentumFromConfig = float(config.get('RunTraining', 'MOMENTUM'))


# ЗАПУСК ТРЕНИРОВКИ #

#: Создание файла для хранения лучшей модели.
createBestModelFile()

#: Извлечение информации о лучшей модели и ее потерях.
bestLoss, bestModelPath = retrieveBestModelInfo()

#: Временная метка для уникального идентификатора обучения.
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

#: Создается объект SummaryWriter для записи логов TensorBoard.
writer = SummaryWriter('Training/runs/fashion_trainer_{}'.format(timestamp))

#: Переменная, отслеживающая текущее количество эпох.
epoch_number = 0

#: Создается модель. 
model = Model()

trainingType = input("Продолжить обучение лучшей модели?\n").lower()
if trainingType in ['yes', 'y', 'ye']:
    # Загружаются параметры лучшей модели.
    f = open("../Training/savedModels/bestModel.txt", "r")
    firstLine = float(f.readline())
    best_model_path = f.readline()
    f.close()
    model.load_state_dict(torch.load(best_model_path))

#: Создаются функция потерь и оптимизатор.
#: Функция потерь - CrossEntropyLoss().
#: Оптимизатор - SGD с заданной скоростью обучения и моментом.
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, momentum=momentumFromConfig)

#: Определяется устройство (GPU или CPU), на котором будет производиться обучение, 
#: и модель перемещается на это устройство.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

#: Переменная для отслеживания наименьших потерь на валидационном наборе данных.
best_vloss = 1_000_000.

#: Цикл по всем эпохам обучения.
for epoch in tqdm(range(epochs)):

    #: Каждую 5-ую эпоху выводится номер текущей эпохи.
    if (epoch_number % 5 == 0):
        print('EPOCH {}:'.format(epoch_number + 1))

    #: Модель переводится в режим обучения (model.train(True)) 
    #: и происходит одна эпоха обучения с помощью функции train_one_epoch(), 
    #: в которой вычисляется среднее значение потерь на тренировочном наборе данных.
    model.train(True)
    avg_loss = train_one_epoch(model, optimizer, loss_fn, epoch_number, writer)

    #: Модель переводится в режим оценки (model.eval()), 
    #: где отключается отслеживание градиентов и используются средние значения 
    #: и стандартное отклонение для пакетной нормализации.
    running_vloss = 0.0
    model.eval()

    #: Оценка модели на валидационном наборе данных. Градиенты не вычисляются (with torch.no_grad()), 
    #: вычисляется среднее значение потерь на валидационном наборе данных.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)

            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    #: Вычисление среднего значения потерь на валидационном наборе данных за текущую эпоху.
    #: Каждая потеря на каждом пакете суммируется в running_vloss, а затем делится на количество пакетов,
    #: чтобы получить среднее значение потерь.
    avg_vloss = running_vloss / (i + 1)

    #: Каждую 5-ую эпоху выводятся значения потерь 
    #: на тренировочном и валидационном наборах данных.
    if epoch_number % 5 == 0:
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    #: Добавляет средние текущие потери как для обучения, так и для валидации в файл журнала.
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Отслеживает лучший результат (по потерям) и сохраняет новую лучшую модель.
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss

        if (bestLoss > best_vloss): # Если у данной модели потери лучше, чем у всех предыдущих моделей, то сохраняем её.
            model_path = 'Training/savedModels/model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)
            saveBestModel(best_vloss, model_path)

    #: Переход к следующей эпохе обучения.
    epoch_number += 1

#: После завершения всех эпох выводится лучшее значение потерь на валидации для всех моделей.
print("\n\nBEST VALIDATION LOSS FOR ALL MODELS: ", bestLoss)
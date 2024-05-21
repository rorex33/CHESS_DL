import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import chess
from tqdm import tqdm
from gym_chess.alphazero.move_encoding import utils
from pathlib import Path
from typing import Optional
import configparser

from TrainingEncodDecod import *

# ПАРСИНГ КОНФИГА #
config = configparser.ConfigParser()
config.readfp(open(r'../config.txt'))

rawDataPath = config.get('GENERAL', 'RAW_DATA_PATH')
preparedDataPath = config.get('GENERAL', 'PREPARED_DATA_PATH')

fractionOfData= int(config.get('MainTraining', 'FRACTION_OF_DATA'))
batchSize = int(config.get('MainTraining', 'BATCH_SIZE'))

## DATASET ##

# ЗАГРУЗКА ДАННЫХ ОБУЧЕНИЯ #

#: Списки, которые будут содержать все ходы и позиции соответственно.
allMoves = []
allBoards = []

#: Получаем список файлов с готовыми данными
#: и определяем их количество (для ходов и позиций).
files = os.listdir(f'{preparedDataPath}')
numOfEach = len(files) // 2 # Половина - ходы, другая половина - позиции

#: Перебираем файлы ходов и позиций.
for i in range(numOfEach):
    try:
        #: Загружаем файл ходов и файл позиций.
        moves = np.load(f"{preparedDataPath}/moves{i}.npy", allow_pickle=True)
        boards = np.load(f"{preparedDataPath}/positions{i}.npy", allow_pickle=True)

        #: Если количество ходов не равно количеству позиций, 
        #: выводится сообщение об ошибке, указывающее на номер набора данных (i).
        if (len(moves) != len(boards)):
            print("ERROR ON i = " + i + "LenMoves: " + len(moves) + "LenBoards:" + len(boards))

        #: Добавляет загруженные ходы и позиции в общие списки.
        allMoves.extend(moves)
        allBoards.extend(boards)
    except:
        print("error: could not load ", i, ", but is still going")

#: Выбирает только долю данных, указанную переменной fractionOfData.
allMoves = np.array(allMoves)[:(int(len(allMoves) * fractionOfData))]
allBoards = np.array(allBoards)[:(int(len(allBoards) * fractionOfData))]

#: Проверяет, что количество ходов и позиций одинаково.
assert len(allMoves) == len(allBoards), "MUST BE OF SAME LENGTH"

#: Эта строка используется для определения индекса, 
#: до которого будет использоваться обучающий набор данных.
#: т.е разделяет данные на обучающий набор (первые 80%) и тестовый набор (последние 20%).
trainDataIdx = int(len(allMoves) * 0.8)


# ПЕРЕНОС ДАННЫХ НА GPU #

#: Определяет устройство (GPU или CPU) для обработки данных.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#: Переносит данные на указанное устройство.
allBoards = torch.from_numpy(np.asarray(allBoards)).to(device)
allMoves = torch.from_numpy(np.asarray(allMoves)).to(device)


# СОХРАНЕНИЕ ЗАГРУЗЧИКОВ ДАННЫХ #

#: Объекты TensorDataset, содержащие обучающие и тестовые данные соответственно.
training_set = torch.utils.data.TensorDataset(allBoards[:trainDataIdx], allMoves[:trainDataIdx])
test_set = torch.utils.data.TensorDataset(allBoards[trainDataIdx:], allMoves[trainDataIdx:])

#: Это загрузчики данных, которые будут использоваться для итерации по обучающим и тестовым данным.
#: Параметр shuffle=True указывает на необходимость перетасовки данных только в обучающем наборе данных.
training_loader = torch.utils.data.DataLoader(training_set, batch_size = batchSize, shuffle=True)
validation_loader = torch.utils.data.DataLoader(test_set, batch_size = batchSize, shuffle=False)


# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ОБУЧЕНИЯ #

def train_one_epoch(model, optimizer, loss_fn, epoch_index, tb_writer):
    """
    Используется для обучения модели на одной эпохе.
    """

    #: running_loss используется для отслеживания общей потери в течение эпохи.
    #: last_loss будет хранить среднюю потерю на пакет на конец эпохи.
    running_loss = 0.
    last_loss = 0.

    #: Перебора данных в training_loader.
    #: используем enumerate(training_loader), 
    #: чтобы получить индексы пакетов, что позволяет нам отслеживать прогресс обучения.
    for i, data in enumerate(training_loader):

        #: Извлекаем входные данные и соответствующие метки из текущего пакета данных.
        inputs, labels = data

        #: обнуляем градиенты перед каждым проходом обратного распространения, 
        #: чтобы избежать накопления градиентов из предыдущих пакетов.
        optimizer.zero_grad()

        #: Делаем прогноз с помощью модели, передавая ей входные данные.
        outputs = model(inputs)

        #: вычисляем потери и их градиенты, 
        #: а затем используем метод backward() для вычисления
        #: градиентов всех параметров модели по отношению к потерям.
        loss = loss_fn(outputs, labels)
        loss.backward()

        #: Обновляем веса модели.
        optimizer.step()

        #: Добавляем текущие потери к переменной running_loss, 
        #: чтобы вычислить средние потери на пакет в конце эпохи.
        running_loss += loss.item()

        #: Если достигнут каждый 1000-ый пакет:
        if i % 1000 == 999:
            #: Средние потери на пакет.
            last_loss = running_loss / 1000

            #: Записываем их в TensorBoard для визуализации. 
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)

            #: сбрасываем running_loss обратно в 0, 
            #: чтобы начать подсчет потерь для следующего набора пакетов.
            running_loss = 0.

    #: Возвращаем последние потери на пакет в текущей эпохе.
    return last_loss


# ФУНКЦИИ ДЛЯ СОХРАНЕНИЯ ЛУЧШЕЙ МОДЕЛИ #

def createBestModelFile():
    """
    создает файл "bestModel.txt", 
    который будет использоваться для хранения информации о лучшей модели.
    """

    #: Проверяет, если папка с сохранёнными моделями.
    #: Если нет, то создаёт её.
    folderPath = Path('./savedModels')
    if (not folderPath.exists()):
        os.mkdir(folderPath)

    #: Создает объект path, который представляет собой путь к файлу с лучшей моделью.
    path = Path('./savedModels/bestModel.txt')

    #:  Если файл не существует, мы создаем его.
    if (not path.exists()):
        f = open(path, "w")

        #: Затем мы записываем в файл строку "10000000", 
        #: которая представляет собой большое число, 
        #: чтобы его можно было перезаписывать с меньшими потерями.
        f.write("10000000") 

        #: записываем вторую строку "testPath", 
        #: которая является пути к тестовой модели 
        #: (может быть изменено на путь к реальной модели после ее обучения).
        f.write("\ntestPath")
        f.close()

def saveBestModel(vloss, pathToBestModel):
    """
    Функция используется для сохранения информации о лучшей модели в файле bestModel.txt.
    """
    #: Открываем файл для записи.
    f = open("../Training/savedModels/bestModel.txt", "w")

    #: Записываем средние потери.
    #: Для этого преобразуем тензор PyTorch в строку
    #: (через str() и item).
    f.write(str(vloss.item()))

    f.write("\n")
    #: Записываем путь к лучшей модели.
    f.write(pathToBestModel)

    #: Сообщение о новой лучшей моделе и её потерях.
    print("NEW BEST MODEL FOUND WITH LOSS:", vloss)

def retrieveBestModelInfo():
    """
    Функция используется для извлечения информации о лучшей модели из файла bestModel.txt.
    """
     #: Открываем файл для чтения.
    f = open('./savedModels/bestModel.txt', "r")

    #: Считываем первую строку (средние потери).
    bestLoss = float(f.readline())

    #: Считываем вторую строку (путь к лучшей модели).
    bestModelPath = f.readline()
    f.close()

    #: Вовзращаем лучшие потери и путь к данной модели.
    return bestLoss, bestModelPath
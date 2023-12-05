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

from TrainingEncodDecod import *

FRACTION_OF_DATA = 1
BATCH_SIZE = 128

#dataset

# Загрузка данных обучения

allMoves = []
allBoards = []

files = os.listdir('./data/preparedData')
numOfEach = len(files) // 2 # Половина - ходы, другая половина - позиции

for i in range(numOfEach):
    try:
        moves = np.load(f"./data/preparedData/moves{i}.npy", allow_pickle=True)
        boards = np.load(f"./data/preparedData/positions{i}.npy", allow_pickle=True)
        if (len(moves) != len(boards)):
            print("ERROR ON i = ", i, len(moves), len(boards))
        allMoves.extend(moves)
        allBoards.extend(boards)
    except:
        print("error: could not load ", i, ", but is still going")

allMoves = np.array(allMoves)[:(int(len(allMoves) * FRACTION_OF_DATA))]
allBoards = np.array(allBoards)[:(int(len(allBoards) * FRACTION_OF_DATA))]
assert len(allMoves) == len(allBoards), "MUST BE OF SAME LENGTH"

# Выровнять доски
# allBoards = allBoards.reshape(allBoards.shape[0], -1)

trainDataIdx = int(len(allMoves) * 0.8)

#NOTE Перенести все данные на GPU, если он доступен
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
allBoards = torch.from_numpy(np.asarray(allBoards)).to(device)
allMoves = torch.from_numpy(np.asarray(allMoves)).to(device)

training_set = torch.utils.data.TensorDataset(allBoards[:trainDataIdx], allMoves[:trainDataIdx])
test_set = torch.utils.data.TensorDataset(allBoards[trainDataIdx:], allMoves[trainDataIdx:])
# Создаваёт загрузчики данных для наших наборов данных; перетасовывает их для обучения, а не для проверки

training_loader = torch.utils.data.DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
     
# Вспомогательные функции для обучения
def train_one_epoch(model, optimizer, loss_fn, epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Здесь мы используем enumerate(training_loader) вместо
    # iter(training_loader) чтобы отслеживать индекс пакета
    # и делать некоторые отчеты внутри эпохи.
    for i, data in enumerate(training_loader):

        # Каждый экземпляр данных представляет собой пару (ввод + метки).
        inputs, labels = data

        # Обнуление градиентов для каждой партии
        optimizer.zero_grad()

        # Сделать прогноз для данной партии
        outputs = model(inputs)

        # Вычислите потери и их градиенты
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Отрегулировать веса обучения
        optimizer.step()

        # Соберать данные и отправить отчет
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # потери на партию
            # print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# 3 функции ниже помогут сохранить лучшую модель
def createBestModelFile():
    # Сначала найти лучшую модель, если она существует:
    folderPath = Path('Training/savedModels')
    if (not folderPath.exists()):
        os.mkdir(folderPath)

    path = Path('Training/savedModels/bestModel.txt')

    if (not path.exists()):
        # Создать файлы
        f = open(path, "w")
        f.write("10000000") # ставим большое число, чтобы оно перезаписывалось с меньшими потерями.
        f.write("\ntestPath")
        f.close()

def saveBestModel(vloss, pathToBestModel):
    f = open("Training/savedModels/bestModel.txt", "w")
    f.write(str(vloss.item()))
    f.write("\n")
    f.write(pathToBestModel)
    print("NEW BEST MODEL FOUND WITH LOSS:", vloss)

def retrieveBestModelInfo():
    f = open('Training/savedModels/bestModel.txt', "r")
    bestLoss = float(f.readline())
    bestModelPath = f.readline()
    f.close()
    return bestLoss, bestModelPath
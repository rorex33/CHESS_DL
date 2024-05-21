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

# МОДЕЛЬ НЕЙРОННЙО СЕТИ #

class Model(torch.nn.Module):

    def __init__(self):
        """Метод инициализации класса."""
        super(Model, self).__init__()

        #: Определяет размер входных данных нейронной сети.
        self.INPUT_SIZE = 896 

        #: Определяет размер выходных данных нейронной сети, 
        #: то есть количество уникальных ходов (пространство действия).
        self.OUTPUT_SIZE = 4672

        #: Инициализирует функцию активации ReLU, 
        #: которая будет использоваться после каждого линейного слоя.
        self.activation = torch.nn.ReLU()

        #: Инициализирует слой Dropout с вероятностью 0.5
        self.dropout = torch.nn.Dropout(p=0.5)

        #: Определяют линейные слои нейронной сети.
        self.linear1 = torch.nn.Linear(self.INPUT_SIZE, 128)
        self.linear2 = torch.nn.Linear(128, 128)
        self.linear3 = torch.nn.Linear(128, 128)
        self.linear4 = torch.nn.Linear(128, 64)
        self.linear5 = torch.nn.Linear(64, self.OUTPUT_SIZE)

        #: Softmax используется для получения вероятностного распределения по выходам модели.
        self.softmax = torch.nn.Softmax(1)
 
    def forward(self, x): #x.shape = (batch size, 896)
        """
        Прямой проход (forward pass) модели. 
        Принимает входные данные x и возвращает выходные данные модели.
        """
        #: Преобразует тип данных входных данных в float32.
        x = x.to(torch.float32)

        #: При необходимости преобразует входные данные в двумерный массив 
        #: (происходит "разглаживание").
        x = x.reshape(x.shape[0], -1)

        #: Применяет линейные слои, их функцию активации и dropout входным данным последовательно.
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.linear3(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.linear4(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.linear5(x)

        return x

    def predict(self, board : chess.Board):
        """
        Метод для прогнозирования следующего хода на основе текущего состояния доски.
        Принимает объект доски board.
        """

        #: Данный контекст отключает вычисления градиентов PyTorch.
        #: Благодаря этому мы не будем обновлять 
        #: веса модели во время выполнения прогнозов.
        with torch.no_grad():

            #: Кодируем текущее состояние доски.
            encodedBoard = encodeBoard(board)

            #: Кодированная доска преобразуется в тензор PyTorch с помощью функции torch.from_numpy(). 
            #: Мы изменяем форму тензора на (1, -1), чтобы преобразовать его в одномерный массив.
            #: Затем мы преобразуем тип данных тензора в float.
            encodedBoard = torch.from_numpy(encodedBoard.reshape(1, -1)).float()

            #: Пропускаем доску через прямой обход нашей модели.
            res = self.forward(encodedBoard)

            #: Отправляем результат на слой softmax,
            #: чтобы получить вероятностное распределение по возможным ходам. 
            #: Это позволяет нам оценить вероятность каждого хода.
            probs = self.softmax(res)

            #: Для удобства уменьшаем тензор до одной оси
            #: и преобразуем его в массив NumPy.
            probs = probs.squeeze().numpy()

            max_attempts = 100
            attempts = 0
            while len(probs) > 0 and attempts < max_attempts: # Попробуем не более max_attempts раз

                #: Находим индекс хода с наибольшей вероятностью в массиве probs.
                moveIdx = probs.argmax()

                #: Декодируем индекс хода в его строковое представление.
                uciMove = decodeMove(moveIdx, board)

                #: Если декодирование не удалось (например, если ход недопустим), 
                #: мы удаляем ход из массива probs и продолжаем цикл.
                if uciMove is None:
                    probs = np.delete(probs, moveIdx)
                    continue

                #: Проверяем, является ли ход допустимым на текущей доске. 
                #: Если да, мы возвращаем этот ход.
                move = chess.Move.from_uci(str(uciMove))
                if move in board.legal_moves:
                    return move
                
                #: Если ход не является допустимым, мы удаляем его из массива probs 
                #: и увеличиваем счетчик попыток.
                probs = np.delete(probs, moveIdx)
                attempts += 1

        return None # Если не найдено возможных ходов, вернуть None
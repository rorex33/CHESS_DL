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

# Модель нейронной сети

class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.INPUT_SIZE = 896 
        # self.INPUT_SIZE = 7*7*13 #NOTE изменение размера ввода для использования cnn (в будущем)
        self.OUTPUT_SIZE = 4672 # = количество уникальных ходов (пространство действия)
        
        # Нужно попробовать внести сюда CNN и пулинг (и узнать, что это)

        # Входная форма для выборки: (8,8,14), сведенная до одномерного массива размером 896.
        # self.cnn1 = nn.Conv3d(4,4,(2,2,4), padding=(0,0,1))
        self.activation = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(self.INPUT_SIZE, 1000)
        self.linear2 = torch.nn.Linear(1000, 1000)
        self.linear3 = torch.nn.Linear(1000, 1000)
        self.linear4 = torch.nn.Linear(1000, 200)
        self.linear5 = torch.nn.Linear(200, self.OUTPUT_SIZE)
        self.softmax = torch.nn.Softmax(1) # Использовать softmax в качестве пробы для каждого хода, dim 1, 
        #поскольку dim 0 — размер пакета
 
    def forward(self, x): #x.shape = (batch size, 896)
        x = x.to(torch.float32)
        # x = self.cnn1(x) # Для использования CNN
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear4(x)
        x = self.activation(x)
        x = self.linear5(x)
        # x = self.softmax(x) # Здесь softmax лучше не юзать, прочитал на stack overflow
        return x

    def predict(self, board : chess.Board):
        """takes in a chess board and returns a chess.move object. NOTE: this function should definitely be written better, but it works for now"""
        with torch.no_grad():
            encodedBoard = encodeBoard(board)
            encodedBoard = encodedBoard.reshape(1, -1)
            encodedBoard = torch.from_numpy(encodedBoard)
            res = self.forward(encodedBoard)
            probs = self.softmax(res)

            probs = probs.numpy()[0] # Тензор больше не нужен, 0, так как это двумерный массив с 1 строкой

            # Перед возвратом убедиться, что ход возможен и может быть декодирован перед возвратом
            while len(probs) > 0: # попробовать 100 раз (максимум), если не выдаёт ошибку
                moveIdx = probs.argmax()
                try:
                    uciMove = decodeMove(moveIdx, board)
                    if (uciMove is None): # Если не смог декодировать
                        probs = np.delete(probs, moveIdx)
                        continue
                    move = chess.Move.from_uci(str(uciMove))
                    if (move in board.legal_moves): # Если ход возможен - возврат, иначе - удалить ход и продолжить цикл
                        return move 
                except:
                    pass
                probs = np.delete(probs, moveIdx) 
                # Реализация не самая лучшая, возможны варианты улучшения:
                # 1) удалить ход, чтобы он не выбирался снова в следующей итерации
                # 2) Возвращать случайный ход
            return None # Если не найдено возможных ходов, вернуть None
   
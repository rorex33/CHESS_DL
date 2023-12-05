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

from Model import Model

saved_model = Model()

# Загрузить лучшую модель
f = open("./Training/savedModels/bestModel.txt", "r")
bestLoss = float(f.readline())
model_path = f.readline()
f.close()

saved_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Запустить игру
board = chess.Board()

while(True):
    print("Введите ход:")
    moveStr = input()
    move = chess.Move.from_uci(moveStr)
    board.push(move)

    # Заставить ИИ сделать ход
    aiMove = saved_model.predict(board)
    print("Ход ИИ:")
    print(aiMove)
    board.push(aiMove)
    print("Доска:")
    print(board)
    print()
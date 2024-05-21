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

# Инициализация и загрузка модели
saved_model = Model()

# Загрузка лучшей модели
with open("./savedModels/bestModel.txt", "r") as f:
    bestLoss = float(f.readline())
    model_path = f.readline().strip()

saved_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Запуск игры
board = chess.Board()

def print_board(board):
    print(board)
    print()

def check_game_over(board):
    if board.is_checkmate():
        print("Мат! Игра завершена.")
        return True
    if board.is_stalemate():
        print("Пат! Игра завершена.")
        return True
    if board.is_insufficient_material():
        print("Недостаточно материала для победы. Игра завершена.")
        return True
    if board.is_seventyfive_moves():
        print("Правило 75 ходов! Игра завершена.")
        return True
    if board.is_fivefold_repetition():
        print("Пятикратное повторение позиции! Игра завершена.")
        return True
    if board.is_variant_draw():
        print("Ничья по правилам варианта! Игра завершена.")
        return True
    return False

while not board.is_game_over():
    # Ход игрока
    print("Введите ход:")
    moveStr = input()
    move = chess.Move.from_uci(moveStr)
    if move in board.legal_moves:
        board.push(move)
        print_board(board)
    else:
        print("Недопустимый ход. Попробуйте снова.")
        continue

    if check_game_over(board):
        break

    # Ход ИИ
    aiMove = None
    max_attempts = 100
    attempts = 0
    
    while aiMove is None and attempts < max_attempts:
        try:
            aiMove = saved_model.predict(board)
            if aiMove is not None and aiMove in board.legal_moves:
                break
            aiMove = None
        except Exception as e:
            print(f"Ошибка при попытке предсказать ход ИИ: {e}")
            aiMove = None
        attempts += 1
    
    if aiMove is None:
        print("ИИ не смог найти допустимый ход. Игра завершена.")
        break

    print("Ход ИИ:")
    print(aiMove)
    board.push(aiMove)
    print_board(board)

    if check_game_over(board):
        break

print("Игра завершена.")

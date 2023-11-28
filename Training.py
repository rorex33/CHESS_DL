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

FRACTION_OF_DATA = 1
BATCH_SIZE = 4



# Вспомогательный метод:

# Декодирование ходов из idx в uci нотацию
def _decodeKnight(action: int) -> Optional[chess.Move]:
    _NUM_TYPES: int = 8

    #: Начальная точка хода коня находится в последнем измерении массива действий 8 x 8 x 73
    _TYPE_OFFSET: int = 56

    #: Набор возможных направлений хода коня, закодированный как
    #: (delta rank, delta square).
    _DIRECTIONS = utils.IndexedTuple(
        (+2, +1),
        (+1, +2),
        (-1, +2),
        (-2, +1),
        (-2, -1),
        (-1, -2),
        (+1, -2),
        (+2, -1),
    )

    from_rank, from_file, move_type = np.unravel_index(action, (8, 8, 73))

    is_knight_move = (
        _TYPE_OFFSET <= move_type
        and move_type < _TYPE_OFFSET + _NUM_TYPES
    )

    if not is_knight_move:
        return None

    knight_move_type = move_type - _TYPE_OFFSET

    delta_rank, delta_file = _DIRECTIONS[knight_move_type]

    to_rank = from_rank + delta_rank
    to_file = from_file + delta_file

    move = utils.pack(from_rank, from_file, to_rank, to_file)
    return move

def _decodeQueen(action: int) -> Optional[chess.Move]:

    _NUM_TYPES: int = 56 # = 8 направлений * 7 (максимальное расстояние до квадрата)

    #: Набор возможных направлений хода ферзя, закодированный как
    #: (delta rank, delta square).
    _DIRECTIONS = utils.IndexedTuple(
        (+1,  0),
        (+1, +1),
        ( 0, +1),
        (-1, +1),
        (-1,  0),
        (-1, -1),
        ( 0, -1),
        (+1, -1),
    )
    from_rank, from_file, move_type = np.unravel_index(action, (8, 8, 73))
    
    is_queen_move = move_type < _NUM_TYPES

    if not is_queen_move:
        return None

    direction_idx, distance_idx = np.unravel_index(
        indices=move_type,
        shape=(8,7)
    )

    direction = _DIRECTIONS[direction_idx]
    distance = distance_idx + 1

    delta_rank = direction[0] * distance
    delta_file = direction[1] * distance

    to_rank = from_rank + delta_rank
    to_file = from_file + delta_file

    move = utils.pack(from_rank, from_file, to_rank, to_file)
    return move

def _decodeUnderPromotion(action):
    _NUM_TYPES: int = 9 #  3 направления * 3 типа фигуры (см. ниже)

    #: Начальная точка  слабого превращения в последнем измерении массива действий 8 x 8 x 73.
    _TYPE_OFFSET: int = 64

    #: Набор возможных направлений для слабого превращения, закодированный как дельта-файл.
    _DIRECTIONS = utils.IndexedTuple(
        -1,
        0,
        +1,
    )

    #: Набор возможных типов фигур для слабого превращения
    #: (превращение в ферзя неявно кодируется соответствующим ходом ферзя)
    _PROMOTIONS = utils.IndexedTuple(
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
    )

    from_rank, from_file, move_type = np.unravel_index(action, (8, 8, 73))

    is_underpromotion = (
        _TYPE_OFFSET <= move_type
        and move_type < _TYPE_OFFSET + _NUM_TYPES
    )

    if not is_underpromotion:
        return None

    underpromotion_type = move_type - _TYPE_OFFSET

    direction_idx, promotion_idx = np.unravel_index(
        indices=underpromotion_type,
        shape=(3,3)
    )

    direction = _DIRECTIONS[direction_idx]
    promotion = _PROMOTIONS[promotion_idx]

    to_rank = from_rank + 1
    to_file = from_file + direction

    move = utils.pack(from_rank, from_file, to_rank, to_file)
    move.promotion = promotion

    return move

# основная функция декодирования, приведенные выше — всего лишь вспомогательные функции
def decodeMove(action: int, board) -> chess.Move:
        move = _decodeQueen(action)
        is_queen_move = move is not None

        if not move:
            move = _decodeKnight(action)

        if not move:
            move = _decodeUnderPromotion(action)

        if not move:
            raise ValueError(f"{action} is not a valid action")

        # Действия кодируют ходы с точки зрения текущего игрока.
        # Если это черный игрок, ход необходимо переориентировать.
        turn = board.turn
        
        if turn == False: # Ход чёрных
            move = utils.rotate(move)

        # Перемещение пешки на домашнюю горизонталь противника ходом ферзя 
        # автоматически считается слабым повышением ферзя.
        # Однако, поскольку ходы ферзя не имеют ссылки на доску и, 
        # следовательно, не могут определить, является ли ходовая фигура пешкой,
        # придется добавить эту информацию сюда вручную.
        if is_queen_move:
            to_rank = chess.square_rank(move.to_square)
            is_promoting_move = (
                (to_rank == 7 and turn == True) or 
                (to_rank == 0 and turn == False)
            )

            piece = board.piece_at(move.from_square)
            if piece is None: #NOTE ВОЗМОЖНО ЭТО НЕПРАВИЛЬНО
                return None
            is_pawn = piece.piece_type == chess.PAWN

            if is_pawn and is_promoting_move:
                move.promotion = chess.QUEEN

        return move

def encodeBoard(board: chess.Board) -> np.array:
 """Converts a board to numpy array representation."""

 array = np.zeros((8, 8, 14), dtype=int)

 for square, piece in board.piece_map().items():
  rank, file = chess.square_rank(square), chess.square_file(square)
  piece_type, color = piece.piece_type, piece.color
 
    # Первые 6 плоскостей кодируют фигуры активного игрока. 
    # Следующие 6 - фигуры активного опонента.
    # Данный класс хранит доски, ориентированные на белого игрока.
    # Белые считаются за активного игрока.
  offset = 0 if color == chess.WHITE else 6
  
    # В шахматах типы фигур начинаются с единицы,
	# это необходимо учитывать
  idx = piece_type - 1
 
  array[rank, file, idx + offset] = 1

 # Счетчики повторений
 array[:, :, 12] = board.is_repetition(2)
 array[:, :, 13] = board.is_repetition(3)

 return array


#dataset

# Загрузка данных обучения

allMoves = []
allBoards = []

files = os.listdir('data/preparedData')
numOfEach = len(files) // 2 # Половина - ходы, другая половина - позиции

for i in range(numOfEach):
    try:
        moves = np.load(f"data/preparedData/moves{i}.npy", allow_pickle=True)
        boards = np.load(f"data/preparedData/positions{i}.npy", allow_pickle=True)
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
    folderPath = Path('./savedModels')
    if (not folderPath.exists()):
        os.mkdir(folderPath)

    path = Path('./savedModels/bestModel.txt')

    if (not path.exists()):
        # Создать файлы
        f = open(path, "w")
        f.write("10000000") # ставим большое число, чтобы оно перезаписывалось с меньшими потерями.
        f.write("\ntestPath")
        f.close()

def saveBestModel(vloss, pathToBestModel):
    f = open("./savedModels/bestModel.txt", "w")
    f.write(str(vloss.item()))
    f.write("\n")
    f.write(pathToBestModel)
    print("NEW BEST MODEL FOUND WITH LOSS:", vloss)

def retrieveBestModelInfo():
    f = open('./savedModels/bestModel.txt', "r")
    bestLoss = float(f.readline())
    bestModelPath = f.readline()
    f.close()
    return bestLoss, bestModelPath
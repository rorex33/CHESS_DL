import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import chess
from tqdm import tqdm
from gym_chess.alphazero.move_encoding import utils
from pathlib import Path
from typing import Optional

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


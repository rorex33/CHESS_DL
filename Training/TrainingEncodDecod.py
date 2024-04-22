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

## ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ##

# ДЕКОДИРОВАНИЕ ХОДОВ ИХ IDX В UCI НОТАЦИЮ:

def _decodeKnight(action: int) -> Optional[chess.Move]:
    """
    Принимает на вход одно целочисленное значение (action) и возвращает ход коня в формате UCI. 
    """

    #: Количество различных ходов коня.
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

    #: Раскрутка одномерного индекса action в трехмерный массив с размерностью (8, 8, 73)
    #: 8 x 8 представляют координаты начальной позиции коня, 
    #: а 73 представляют различные типы ходов.
    from_rank, from_file, move_type = np.unravel_index(action, (8, 8, 73))

    #: Эта строка проверяет, является ли тип хода действительным ходом коня, 
    #: основываясь на диапазоне индексов, определенных _TYPE_OFFSET и _NUM_TYPES.
    is_knight_move = (
        _TYPE_OFFSET <= move_type
        and move_type < _TYPE_OFFSET + _NUM_TYPES
    )

    #: Если тип хода не является действительным ходом коня, функция возвращает None.
    if not is_knight_move:
        return None

    #: Вычисоение индекса типа хода коня в пределах диапазона от 0 до _NUM_TYPES.
    knight_move_type = move_type - _TYPE_OFFSET

    #: Извлечение изменения строки и столбца для конкретного типа хода коня, 
    #: используя knight_move_type.
    delta_rank, delta_file = _DIRECTIONS[knight_move_type]

    #: Вычисление конечных координат для хода коня.
    to_rank = from_rank + delta_rank
    to_file = from_file + delta_file

    #: Формирование хода коня, 
    #: используя начальные и конечные координаты, 
    #: (ход в формате UCI).
    move = utils.pack(from_rank, from_file, to_rank, to_file)

    #: Возврат хода.
    return move

def _decodeQueen(action: int) -> Optional[chess.Move]:
    """
    Принимает на вход одно целочисленное значение (action) и возвращает ход ферзя в формате UCI. 
    """

    #: Общее количество различных ходов ферзя.
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

    #: Раскрутка одномерного индекса action в трехмерный массив с размерностью (8, 8, 73)
    #: 8 x 8 представляют координаты начальной позиции ферзя, 
    #: а 73 представляют различные типы ходов.
    from_rank, from_file, move_type = np.unravel_index(action, (8, 8, 73))
    
    #: Эта строка проверяет, является ли тип хода действительным ходом ферзя, 
    #: основываясь на диапазоне индексов, определенных _NUM_TYPES.
    is_queen_move = move_type < _NUM_TYPES

    #: Если тип хода не является действительным ходом ферзя, функция возвращает None.    
    if not is_queen_move:
        return None

    #: Раскрутка индекса move_type в кортеж (direction_idx, distance_idx) 
    #: с размерностью (8, 7), где 8 представляет направления, а 7 представляет расстояния.
    direction_idx, distance_idx = np.unravel_index(
        indices=move_type,
        shape=(8,7)
    )

    #: Извлечение направления (direction) и расстояния (distance) для соответствующего типа хода ферзя.
    direction = _DIRECTIONS[direction_idx]
    distance = distance_idx + 1

    #: Вычисление изменения строки и стобца для хода ферзя,
    #: учитывая выбранное направление и расстояние.
    delta_rank = direction[0] * distance
    delta_file = direction[1] * distance

    #: Вычисление конечных координаты для хода ферзя.
    to_rank = from_rank + delta_rank
    to_file = from_file + delta_file

    #: Формирование хода ферзя, 
    #: используя начальные и конечные координаты, 
    #: (ход в формате UCI).
    move = utils.pack(from_rank, from_file, to_rank, to_file)

    #: Возврат хода.
    return move

def _decodeUnderPromotion(action):
    """
    Принимает на вход одно целочисленное значение (action) и возвращает ход cо слабым превращением пешки. 
    """

    #: Количество различных типов продвинутой пешки.
    _NUM_TYPES: int = 9 #  3 направления * 3 типа фигуры (см. ниже)

    #: Смещение индекса типа продвинутой пешки в многомерном массиве действий.
    _TYPE_OFFSET: int = 64

    #: Набор возможных направлений для слабого превращения, закодированный как изменение столбца.
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

    
    #: Раскрутка индекса move_type в кортеж (direction_idx, distance_idx) 
    #: с размерностью (8, 7), где 8 представляет направления, а 7 представляет расстояния.
    from_rank, from_file, move_type = np.unravel_index(action, (8, 8, 73))

    #: Эта строка проверяет, является ли тип хода действительным ходом для слабого превращения пешки,
    #: основываясь на диапазоне индексов, определенных _TYPE_OFFSET и _NUM_TYPES.
    is_underpromotion = (
        _TYPE_OFFSET <= move_type
        and move_type < _TYPE_OFFSET + _NUM_TYPES
    )

    #: Если тип хода не является действительным для слабого превращения пешки, функция возвращает None.    
    if not is_underpromotion:
        return None

    #: Вычисление индекса типа слабого превращения пешки в пределах диапазона от 0 до _NUM_TYPES.
    underpromotion_type = move_type - _TYPE_OFFSET

    #: Раскуртка индекса underpromotion_type 
    #: в кортеж (direction_idx, promotion_idx) 
    #: с размерностью (3, 3), где 3 представляют направления, 
    #: а также типы фигур для слабого превращения.
    direction_idx, promotion_idx = np.unravel_index(
        indices=underpromotion_type,
        shape=(3,3)
    )

    #: Извлечение направления (direction) и типа превращения (promotion) для соответствующего типа слабого превращения пешки.
    direction = _DIRECTIONS[direction_idx]
    promotion = _PROMOTIONS[promotion_idx]

    #: Вычисление конечных координаты для слабого превращения пешки.
    to_rank = from_rank + 1
    to_file = from_file + direction

    #: Формирование хода для слабого превращения пешки, 
    #: используя начальные и конечные координаты, 
    #: (ход в формате UCI).
    move = utils.pack(from_rank, from_file, to_rank, to_file)
    move.promotion = promotion

    #: Возврат хода.
    return move


## ОСНОВНАЯ ФУНКЦИЯ ДЕКОДИРОВАНИЯ ##

def decodeMove(action: int, board) -> chess.Move:
        """
        Используется для декодирования действия в ход на шахматной доске. 
        Принимает два аргумента: action - действие (ход) в виде целого числа, 
        и board - объект шахматной доски.
        """

        #: Декодирование действия в ход ферзя
        #: и проверка, был ли ход действительным.
        move = _decodeQueen(action)
        is_queen_move = move is not None

        #: Если ход ферзя не действиельный,
        #: то попытка декодирования хода коня.
        if not move:
            move = _decodeKnight(action)

        #: Если ход коня не действиельный,
        #: то попытка декодирования слабого превращения пешки.
        if not move:
            move = _decodeUnderPromotion(action)

        #: Если ни один из предыдущих вариантов не сработал,
        #: то выводим сообщение об ошибке (нелегальный ход).
        if not move:
            raise ValueError(f"{action} is not a valid action")

        #: Получение текущего цвета игрока на доске.
        turn = board.turn
        
        #: сли текущий ход принадлежит чёрным (т.е., turn равен False), 
        #: ход поворачивается с помощью функции utils.rotate(). 
        #: Это делается для корректного представления хода относительно черной стороны доски.
        if turn == False: # Ход чёрных
            move = utils.rotate(move)

        #: Если исходный ход был действительным ходом ферзя:
        if is_queen_move:
            
            #: Вычисляет столбец, на который произойдет ход.
            to_rank = chess.square_rank(move.to_square)

            #: Определяет, является ли ход превращением пешки, исходя из цвета и финальной позиции пешки.
            is_promoting_move = (
                (to_rank == 7 and turn == True) or 
                (to_rank == 0 and turn == False)
            )

            #: Получает фигуру, которая совершает ход.
            piece = board.piece_at(move.from_square)

            #: Проверяет, была ли успешно получена фигура,
            #: которая совершает ход.
            if piece is None:
                return None
            
            #: Проверяет, является ли фигура пешкой.
            is_pawn = piece.piece_type == chess.PAWN

            #: Если фигура является пешкой и ход является ходом превращения, 
            #: устанавливает тип превращения в ферзя.
            if is_pawn and is_promoting_move:
                move.promotion = chess.QUEEN

        #: Возвращает декодированный ход.
        return move

def encodeBoard(board: chess.Board) -> np.array:
    """Преобразует объект шахматной доски в массив numpy."""
    
    #: Создание массива numpy размером 8x8x14, заполненный нулями и типа данных int. 
    #: Этот массив будет использоваться для представления шахматной доски. 
    #: 8x8 - это размеры шахматной доски, а 14 - это количество каналов (или "плоскостей") для каждой клетки доски.
    array = np.zeros((8, 8, 14), dtype=int)

    #: Перебирает все фигуры на доске и их позиции с помощью метода piece_map(), 
    #: который возвращает словарь, содержащий соответствия клеток и фигур.
    for square, piece in board.piece_map().items():

        #: Получение строки и столбца клетки.
        rank, file = chess.square_rank(square), chess.square_file(square)

        #: Получение типа фигуры и ее цвета.
        piece_type, color = piece.piece_type, piece.color
        
        #: Определение смещения для кодирования фигур. 
        #: Если фигура принадлежит белому игроку, 
        #: смещение устанавливается в 0, в противном случае - в 6.
        offset = 0 if color == chess.WHITE else 6
        
        #: Тип фигуры уменьшается на 1, так как типы фигур 
        #: в библиотеке python-chess начинаются с 1, 
        #: а индексы массивов в Python начинаются с 0.
        idx = piece_type - 1
        
        #: В массиве array устанавливается значение 1 в соответствующей позиции для кодирования фигуры.
        array[rank, file, idx + offset] = 1

    #: Эти две строки устанавливают счетчики повторений для клеток доски. 
    #: Первая возвращает матрицу, в которой каждая клетка равна 1, 
    #: если на этой клетке было повторение хода за два последних хода, иначе 0. 
    #: Аналогично для второй, но для трех последних ходов.
    array[:, :, 12] = board.is_repetition(2)
    array[:, :, 13] = board.is_repetition(3)

    #: Возвращает массив array, который представляет шахматную доску в виде массива numpy.
    return array
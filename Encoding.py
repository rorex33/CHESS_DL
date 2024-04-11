import numpy as np
import gym
import chess
import os
import gym.spaces
from gym_chess.alphazero.move_encoding import utils, queenmoves, knightmoves, underpromotions
from typing import List
import configparser

#Парсинг конфига
config = configparser.ConfigParser()
config.readfp(open(r'config.txt'))

rawDataPath = config.get('GENERAL', 'rawDataPath')
preparedDataPath = config.get('GENERAL', 'preparedDataPath')

# Эта строка кода создает среду для обучения и тестирования алгоритма с использованием библиотеки OpenAI Gym.
env = gym.make('ChessAlphaZero-v0')

# КОДИРОВАНИЕ ФУНКЦИЙ ИЗ ALPHA ZERO:

def encodeBoardFromFen(fen: str) -> np.array:
    #: Конвертация доски из строки FEN в массив NumPy.
	board = chess.Board(fen)
	return encodeBoard(board)
	
def encodeBoard(board: chess.Board) -> np.array:
	"""Конвертация объекта доски в массив NumPy."""

    #: Создание трехмерного массива NumPy размером 8x8x14, заполненного нулями. 
    #: Этот массив будет использоваться для кодирования доски.
	array = np.zeros((8, 8, 14), dtype=int)

    #: Цикла по всем фигурам на доске.
    #: piece_map() возвращает словарь, содержащий информацию о фигурах на доске.
	for square, piece in board.piece_map().items():
		rank, file = chess.square_rank(square), chess.square_file(square)
		piece_type, color = piece.piece_type, piece.color
	
        # Первые 6 плоскостей кодируют фигуры активного игрока. 
        # Следующие 6 - фигуры активного опонента.
        # Данный класс хранит доски, ориентированные на белого игрока.
        # Белые считаются за активного игрока.
        #: Определение смещения в массиве для текущего цвета фигуры.
		offset = 0 if color == chess.WHITE else 6
		
		#: Вычисление индекса типа фигуры в массиве. Типы фигур в 
		# шахматах начинаются с единицы, поэтому вычитается 1.
		idx = piece_type - 1

        #: Установка соответствующего элемента массива в 1, 
        #: чтобы закодировать наличие фигуры на данной позиции.
		array[rank, file, idx + offset] = 1

	#: Счетчики повторений
	array[:, :, 12] = board.is_repetition(2) # повторение позиции после двух ходов.
	array[:, :, 13] = board.is_repetition(3) # повторение позиции после Трех ходов.

    #: Возврат закодированной доски в виде массива NumPy.
	return array


def decodeMove(move: int):
    """Функция декодирует целочисленный ход в его фактическое представление (в рамках среды ChessAlphaZero)."""
    return env.decode(move)

# ИЗМЕНЕННЫЕ ФУКЦИИ КОДИРОВАНИЯ ИЗ OPENAI

def encodeKnight(move: chess.Move):
    """Кодирует ходы коня."""

    # Количество различных ходов коня.
    _NUM_TYPES: int = 8

    #: Начальная точка хода коня находится в последнем измерении массива действий 8 x 8 x 7.
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

    #: Распаковка координат начальной и конечной позиций хода коня из объекта move.
    from_rank, from_file, to_rank, to_file = utils.unpack(move)

    #: Вычисление разности координат между начальной и конечной позициями хода коня.
    delta = (to_rank - from_rank, to_file - from_file)

    #: Проверка, является ли заданное направление движения коня допустимым.
    is_knight_move = delta in _DIRECTIONS
    
    if not is_knight_move:
        return None

    #: Определение типа хода коня, основанного на индексе направления в структуре _DIRECTIONS.
    knight_move_type = _DIRECTIONS.index(delta)
    #: Вычисление типа хода с учетом смещения.
    move_type = _TYPE_OFFSET + knight_move_type

    #: преобразования трехмерных координат (строка, столбец, тип хода) 
    #: в одномерный индекс в соответствии с заданными размерностями.
    action = np.ravel_multi_index(
        multi_index=((from_rank, from_file, move_type)),
        dims=(8, 8, 73)
    )

    #: Возвращение закодированного хода коня.
    return action


def encodeQueen(move: chess.Move):
    """Кодирует ходы ферзя."""

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

    #: Распаковка координат начальной и конечной позиций хода ферзя из объекта move.
    from_rank, from_file, to_rank, to_file = utils.unpack(move)

    #: Вычисление разности координат между начальной и конечной позициями хода ферзя.
    delta = (to_rank - from_rank, to_file - from_file)

    #: Проверки, является ли движение ферзя горизонтальным / вертикальным / диагональным.
    is_horizontal = delta[0] == 0
    is_vertical = delta[1] == 0
    is_diagonal = abs(delta[0]) == abs(delta[1])

    #: Проверка, является ли ферзь продвинутой пешкой или нет.
    is_queen_move_promotion = move.promotion in (chess.QUEEN, None)

    #: Движение по горизонтали, вертикали или диагонали, и не является ли это продвинутой пешкой.
    is_queen_move = (
        (is_horizontal or is_vertical or is_diagonal) 
            and is_queen_move_promotion
    )

    if not is_queen_move:
        return None

    #: Вычисление направления движения ферзя в виде кортежа из знаков разностей координат.
    direction = tuple(np.sign(delta))

    #: Вычисление максимального значения абсолютной разности координат, 
    #: что представляет собой максимальное расстояние движения ферзя.
    distance = np.max(np.abs(delta))

    #: Определение индекса направления движения в структуре _DIRECTIONS.
    direction_idx = _DIRECTIONS.index(direction)
    #: Вычисление индекса расстояния движения ферзя.
    distance_idx = distance - 1

    #: Преобразования двухмерных координат (направление, расстояние) 
    #: в одномерный индекс в соответствии с заданными размерностями.
    move_type = np.ravel_multi_index(
        multi_index=([direction_idx, distance_idx]),
        dims=(8,7)
    )

    #: Преобразования трехмерных координат (строка, столбец, тип хода) 
    #: в одномерный индекс в соответствии с заданными размерностями.
    action = np.ravel_multi_index(
        multi_index=((from_rank, from_file, move_type)),
        dims=(8, 8, 73)
    )

    
    #: Возвращение закодированного хода ферзя.
    return action


def encodeUnder(move):
    """Кодирует превращение пешки."""

    #: Количество различных типов продвинутой пешки.
    _NUM_TYPES: int = 9 # = 3 направления * 3 типа фигуры

    #: Смещение индекса типа продвинутой пешки в многомерном массиве действий.
    _TYPE_OFFSET: int = 64

    #: Набор возможных направлений хода пешки.
    _DIRECTIONS = utils.IndexedTuple(
        -1,
        0,
        +1,
    )

    #: Типы фигур, в которые может превратиться пешка.
    _PROMOTIONS = utils.IndexedTuple(
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
    )

    #: Распаковка координат начальной и конечной позиций хода ферзя из объекта move.
    from_rank, from_file, to_rank, to_file = utils.unpack(move)

    #: Проверка, является ли ход продвинутой пешкой 
    #: и подходит ли он для превращения пешки на следующем ходу.
    is_underpromotion = (
        move.promotion in _PROMOTIONS 
        and from_rank == 6 
        and to_rank == 7
    )

    if not is_underpromotion:
        return None
    
    #: Вычисление разности координат стобцов начальной и конечной позиций хода.
    delta_file = to_file - from_file

    #: Определение индекса направления движения пешки в структуре _DIRECTIONS.
    direction_idx = _DIRECTIONS.index(delta_file)

    #: Определение индекса типа фигуры, 
    #: в которую будет превращена пешка, в структуре _PROMOTIONS.
    promotion_idx = _PROMOTIONS.index(move.promotion)

    #: Преобразование двухмерных координат (направление, тип фигуры) 
    #: в одномерный индекс в соответствии с заданными размерностями.
    underpromotion_type = np.ravel_multi_index(
        multi_index=([direction_idx, promotion_idx]),
        dims=(3,3)
    )

    #: Вычисление типа хода с учетом смещения.
    move_type = _TYPE_OFFSET + underpromotion_type

    #: Преобразование трехмерных координат (строка, столбец, тип хода) 
    #: в одномерный индекс в соответствии с заданными размерностями.
    action = np.ravel_multi_index(
        multi_index=((from_rank, from_file, move_type)),
        dims=(8, 8, 73)
    )

    #: Возвращение закодированного хода продвинутой пешки.
    return action

def encodeMove(move: str, board) -> int:
    """Кодирование ходов в соответствии с текущим состоянием доски."""

    #: Преобразовнаие строчной версии хода в объект типа chess.move
    move = chess.Move.from_uci(move)

    #: Если ход чёрных, то ход "переворачивается"
    if board.turn == chess.BLACK:
        move = utils.rotate(move)

    #: Действием по умолчанию является ход ферзём.
    action = encodeQueen(move)

    #: Если ход ферзём не возможен, то действием становиться ход конём.
    if action is None:
        action = encodeKnight(move)

    #: Если ходы конём и ферзём не возможны, то действием становиться ход продвинутой пешкой.
    if action is None:
        action = encodeUnder(move)

    #: Если ни один из данных ходов не возможен, то выдаём сообщение об ошибке.
    if action is None:
        raise ValueError(f"{move} is not a valid move")

    #: Возвращается закодированный ход.
    return action

def encodeAllMovesAndPositions():
    """Функция для кодирования всех ходов и позиций из папки rawData."""

    #: Объект используется для отслеживания текущего состояния доски 
    #: и определения, чей сейчас ход.
    board = chess.Board()

    #: В начале ход передаётся чёрным, это измениться при первом запуске.
    board.turn = False

    # Получение списка файлов в папке:
    files = os.listdir(f'{rawDataPath}')

    #: Цикл по всем файлам в папке.
    for idx, f in enumerate(files):

        #: Переменные для извлечения ходов и позиций из файла
        movesAndPositions = np.load(f'{rawDataPath}/{f}', allow_pickle=True)
        moves = movesAndPositions[:,0]
        positions = movesAndPositions[:,1]

        #: Инициализация пустых списков для хранения закодированных ходов и позиций.
        encodedMoves = []
        encodedPositions = []

        # Цикл по всем ходам.
        for i in range(len(moves)):

            #: Переключение очереди хода на противоположного игрока.
            board.turn = (not board.turn)
            try:
                #: Кодировка текущего хода и позиции
                encodedMoves.append(encodeMove(moves[i], board)) 
                encodedPositions.append(encodeBoardFromFen(positions[i]))
            except:
                try:
                    # Сменить ход, так как мы иногда пропускаем ходы, 
                    # возможно, нам придется сменить ход
                    board.turn = (not board.turn)
                    encodedMoves.append(encodeMove(moves[i], board)) 
                    encodedPositions.append(encodeBoardFromFen(positions[i]))
                except:
                    print(f'error in file: {f}')
                    with open('EncodingErrors.txt', 'a') as w:
                        w.write(f'{f} \n') # запись имени файла, в котором произошла ошибка.
                    print("Turn: ", board.turn)
                    print("Move:", moves[i])
                    print("Position:", positions[i])
                    print("Index:", i)
                    break

        #: Сохранение закодированных ходов и позиций
        #: в файл с именем, зависящим от индекса цикла.    
        np.save(f'{preparedDataPath}/moves{idx}', np.array(encodedMoves))
        np.save(f'{preparedDataPath}/positions{idx}', np.array(encodedPositions))
    


encodeAllMovesAndPositions()

#NOTE: формаn файлов:
#moves: (количество ходов в игре)
#positions: (количество ходов в игре, 8, 8, 14) (количество ходов в игре включает как ходы чёрных, так и ходы белых)
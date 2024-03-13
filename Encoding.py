import numpy as np
import gym
import chess
import os
import gym.spaces
from gym_chess.alphazero.move_encoding import utils, queenmoves, knightmoves, underpromotions
from typing import List

env = gym.make('ChessAlphaZero-v0')

# Кодировачные функции из alpha zero:

def encodeBoardFromFen(fen: str) -> np.array:
	board = chess.Board(fen)
	return encodeBoard(board)
	
def encodeBoard(board: chess.Board) -> np.array:
	"""Конвентирует доску в nympy array представление."""

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


def decodeMove(move: int):
	return env.decode(move)

# Исправление функций кодирования из openai

def encodeKnight(move: chess.Move):
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

    from_rank, from_file, to_rank, to_file = utils.unpack(move)

    delta = (to_rank - from_rank, to_file - from_file)
    is_knight_move = delta in _DIRECTIONS
    
    if not is_knight_move:
        return None

    knight_move_type = _DIRECTIONS.index(delta)
    move_type = _TYPE_OFFSET + knight_move_type

    action = np.ravel_multi_index(
        multi_index=((from_rank, from_file, move_type)),
        dims=(8, 8, 73)
    )

    return action


def encodeQueen(move: chess.Move):
    _NUM_TYPES: int = 56 # = 8 направлений * 7 (максимальное расстояние до квадрата)
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

    from_rank, from_file, to_rank, to_file = utils.unpack(move)

    delta = (to_rank - from_rank, to_file - from_file)

    is_horizontal = delta[0] == 0
    is_vertical = delta[1] == 0
    is_diagonal = abs(delta[0]) == abs(delta[1])
    is_queen_move_promotion = move.promotion in (chess.QUEEN, None)

    is_queen_move = (
        (is_horizontal or is_vertical or is_diagonal) 
            and is_queen_move_promotion
    )

    if not is_queen_move:
        return None

    direction = tuple(np.sign(delta))
    distance = np.max(np.abs(delta))

    direction_idx = _DIRECTIONS.index(direction)
    distance_idx = distance - 1

    move_type = np.ravel_multi_index(
        multi_index=([direction_idx, distance_idx]),
        dims=(8,7)
    )

    action = np.ravel_multi_index(
        multi_index=((from_rank, from_file, move_type)),
        dims=(8, 8, 73)
    )

    return action


def encodeUnder(move):
    _NUM_TYPES: int = 9 # = 3 направления * 3 типа фигуры (см. ниже)
    _TYPE_OFFSET: int = 64
    _DIRECTIONS = utils.IndexedTuple(
        -1,
        0,
        +1,
    )
    _PROMOTIONS = utils.IndexedTuple(
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
    )

    from_rank, from_file, to_rank, to_file = utils.unpack(move)

    is_underpromotion = (
        move.promotion in _PROMOTIONS 
        and from_rank == 6 
        and to_rank == 7
    )

    if not is_underpromotion:
        return None

    delta_file = to_file - from_file

    direction_idx = _DIRECTIONS.index(delta_file)
    promotion_idx = _PROMOTIONS.index(move.promotion)

    underpromotion_type = np.ravel_multi_index(
        multi_index=([direction_idx, promotion_idx]),
        dims=(3,3)
    )

    move_type = _TYPE_OFFSET + underpromotion_type

    action = np.ravel_multi_index(
        multi_index=((from_rank, from_file, move_type)),
        dims=(8, 8, 73)
    )

    return action


def encodeMove(move: str, board) -> int:
    move = chess.Move.from_uci(move)
    if board.turn == chess.BLACK:
        move = utils.rotate(move)

    action = encodeQueen(move)

    if action is None:
        action = encodeKnight(move)

    if action is None:
        action = encodeUnder(move)

    if action is None:
        raise ValueError(f"{move} is not a valid move")

    return action

# Функция для кодирования всех ходов и позиций из папки rawData
def encodeAllMovesAndPositions():
    board = chess.Board() # Это используется для изменения, чья сейчас очередь, чтобы кодирование работало
    board.turn = False # Сначала передать ход чёрным, измениться при первом запуске

    # Найти все файлы в папке:
    files = os.listdir('data/rawData')
    for idx, f in enumerate(files):
        movesAndPositions = np.load(f'data/rawData/{f}', allow_pickle=True)
        moves = movesAndPositions[:,0]
        positions = movesAndPositions[:,1]
        encodedMoves = []
        encodedPositions = []


        for i in range(len(moves)):
            board.turn = (not board.turn) # Поменять местами ходы
            try:
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
                        w.write(f'{f} \n')
                    print("Turn: ", board.turn)
                    print(moves[i])
                    print(positions[i])
                    print(i)
                    break
            
        np.save(f'data/preparedData/moves{idx}', np.array(encodedMoves))
        np.save(f'data/preparedData/positions{idx}', np.array(encodedPositions))
    
encodeAllMovesAndPositions()

#NOTE: формаn файлов:
#moves: (количество ходов в игре)
#positions: (количество ходов в игре, 8, 8, 14) (количество ходов в игре включает как ходы чёрных, так и ходы белых)
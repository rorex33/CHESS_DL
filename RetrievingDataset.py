import chess
from stockfish import Stockfish
stockfish = Stockfish(path=r"stockfish\stockfish-windows-x86-64-avx2")
import random
from pprint import pprint
import numpy as np
import os
import glob
import time
from multiprocessing import Pool
import configparser

# ПАРСИНГ КОНФИГА #
config = configparser.ConfigParser()
config.readfp(open(r'config.txt'))

rawDataPath = config.get('GENERAL', 'RAW_DATA_PATH')
dataPath = config.get('GENERAL', 'DATA_PATH')

gamesPerCore = int(config.get('RetrievingDataset', 'GAMES_PER_CORE'))
for6cores =  [gamesPerCore, gamesPerCore, gamesPerCore, gamesPerCore, gamesPerCore, gamesPerCore]


# ВСПОМОГАТЕЛЬНЫЕ ФУКНЦИИ #

def checkEndCondition(board):
	"""
	Функция для проверки завершающей позиции.
	"""

	#: Это условие проверяет несколько возможных условий, при которых игра может завершиться:
	#: является ли текущая позиция шахоматной (когда король под шахом и не имеет возможности уйти).
	#: является ли текущая позиция патом (когда у игрока нет доступных ходов, 
	#: и его король не находится под шахом).
	#: достаточно ли материала на доске для выигрыша (например, когда на доске остаются только короли).
	#: можно ли претендовать на ничью по правилу пятидесяти ходов без хода пешки и без взятия.
	#: можно ли претендовать на ничью (например, если обе стороны согласились на ничью).
	if (board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition() or board.can_claim_fifty_moves() or board.can_claim_draw()):
		return True
	return False

def saveData(moves, positions):
	'''
    Функция для сохранения ходов и позиций.
    '''

	#: Преобразует список ходов в массив NumPy и изменяет его форму 
 	#: на одномерный массив с одним столбцом.
	moves = np.array(moves).reshape(-1, 1)

	#: Аналогично, преобразует список позиций в массив NumPy.
	positions = np.array(positions).reshape(-1,1)

	#: Объединяет массивы  по столбцам, создавая двумерный массив, 
 	#: где первый столбец содержит ходы, а второй - позиции.
	movesAndPositions = np.concatenate((moves, positions), axis = 1)

	#: Индекс для названия сохраняемого файла.
	nextIdx = findNextIdx()

	#: Сохранение файла.
	np.save(f"{rawDataPath}/movesAndPositions{nextIdx}.npy", movesAndPositions)
	print(f"file {nextIdx} saved successfully")


def runGame(numMoves, filename = "movesAndPositions1.npy"):
	"""
	Функция для запуска сохраненной игры (тестирование).
	"""
	testing = np.load(f"{dataPath}/{filename}")
	moves = testing[:, 0]
	if (numMoves > len(moves)):
		print("Must enter a lower number of moves than maximum game length. Game length here is: ", len(moves))
		return

	testBoard = chess.Board()

	for i in range(numMoves):
		move = moves[i]
		testBoard.push_san(move)
	return testBoard


# ОСНОВНЫЕ ФУНКЦИИ #

def findNextIdx():
	"""
	Функция для поиска индексов существующих файлов. Если файла с таким индексам нет, то возвращает 1. Иначе возвращает индекс + 1. 
	"""

	#: Получение списка уже существующих файлов. 
	files = (glob.glob(f"{rawDataPath}/*.npy"))

	#: Если файлов нет:
	if (len(files) == 0):
		return 1 # Если файла нет, вернуть 1
	
	#: Устанавливает начальное значение переменной равным 0, 
 	#: чтобы начать поиск самого большого индекса.
	highestIdx = 0

	#: Перебирает каждый файл в списке. 
	for f in files:
		file = f

		#: Извлекает индекс из имени файла.
		currIdx = file.split("movesAndPositions")[-1].split(".npy")[0]

		#: Передаёт в highestIdx максимальный индекс из всех файлов.
		highestIdx = max(highestIdx, int(currIdx))

	#: Возвращает наибольший найденный индекс увеличенным на 1, 
 	#: чтобы получить следующий доступный индекс для нового файла.
	return int(highestIdx)+1

def mineGames(numGames : int):
	"""
	Функция для майнинга партий.
	"""

	#: Не продолжать игру при достижении данного количества ходов.
	MAX_MOVES = 500

	#: Счётчик игр в данном пуле (в данной работающей фукнции).
	countOfGames = 1

	for i in range(numGames):

		#: Списки для хранения ходов и позиций текущей партии.
		currentGameMoves = []
		currentGamePositions = []

		#: Cоздание объекта доски для шахматной партии.
		board = chess.Board()

		#: Установка начальной позиции.
		stockfish.set_position([])

		for i in range(MAX_MOVES):
			
			#: Случайно выбирать из данных трёх ходов.
			moves = stockfish.get_top_moves(3)

			#: Если доступно меньше трёх ходов, то выбрать первый, 
   			#: если доступных ходов нет, то завершить партию.
			if (len(moves) == 0):
				print(f"game {countOfGames}/{gamesPerCore} is over")
				break
			elif (len(moves) == 1):
				move = moves[0]["Move"]
			elif (len(moves) == 2):
				move = random.choices(moves, weights=(80, 20), k=1)[0]["Move"]
			else:
				move = random.choices(moves, weights=(80, 15, 5), k=1)[0]["Move"]

			#: Добавление текущей позиции в список позиций партии.
			currentGamePositions.append(stockfish.get_fen_position())

			#: Добавление хода в список ходов партии.
			currentGameMoves.append(move)
			
			#: Конвертирование хода из формата UCI в формат chess.Move.
			move = chess.Move.from_uci(str(move))

			#: Выполнение хода на доске.
			board.push(move)

			#: Обновление позиции.
			stockfish.set_position(currentGameMoves)

			#: Проверка, является ли позиция завершающей. 
   			#: Если да, партия завершается.
			if (checkEndCondition(board)):
				print(f"game {countOfGames}/{gamesPerCore} is over")
				break

		#: Увеличение счетчика игр.
		countOfGames = countOfGames + 1

		#: Сохранения ходов и позиций проведённой партии.
		saveData(currentGameMoves, currentGamePositions)
		
	
# MAIN ФУНКЦИЯ #

if __name__ == '__main__':

	#: Отслеживание временных затрат, начало отсчёта.
	start_time = time.time()

	#: Вызываем функцию на 6 ядер процессора.
	with Pool(6) as p:
		p.map(mineGames, for6cores)

	#: Оотслеживание временных затрат, конец отсчёта.
	end_time = time.time()

	#: Вычисления и вывод временных затрат.
	elapsed_time = end_time - start_time
	print('Elapsed time: ', elapsed_time/60)
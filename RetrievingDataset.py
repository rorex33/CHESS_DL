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

# Вспомогательная функция:
def checkEndCondition(board):
	if (board.is_checkmate() or board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition() or board.can_claim_fifty_moves() or board.can_claim_draw()):
		return True
	return False

def saveData(moves, positions):
	moves = np.array(moves).reshape(-1, 1)
	positions = np.array(positions).reshape(-1,1)
	movesAndPositions = np.concatenate((moves, positions), axis = 1)

	nextIdx = findNextIdx()
	np.save(f"data/rawData/movesAndPositions{nextIdx}.npy", movesAndPositions)
	print("Saved successfully")


def runGame(numMoves, filename = "movesAndPositions1.npy"):
	"""run a game you stored"""
	testing = np.load(f"data/{filename}")
	moves = testing[:, 0]
	if (numMoves > len(moves)):
		print("Must enter a lower number of moves than maximum game length. Game length here is: ", len(moves))
		return

	testBoard = chess.Board()

	for i in range(numMoves):
		move = moves[i]
		testBoard.push_san(move)
	return testBoard

# Сохранение
def findNextIdx():
	files = (glob.glob(r"data\rawData\*.npy"))
	if (len(files) == 0):
		return 1 # Если файла нет, вернуть 1
	highestIdx = 0
	for f in files:
		file = f
		currIdx = file.split("movesAndPositions")[-1].split(".npy")[0]
		highestIdx = max(highestIdx, int(currIdx))

	return int(highestIdx)+1

def mineGames(numGames : int):
	"""mines numGames games of moves"""
	MAX_MOVES = 500 # Не продолжать игру при достижении данного количества ходов

	for i in range(numGames):
		currentGameMoves = []
		currentGamePositions = []
		board = chess.Board()
		stockfish.set_position([])

		for i in range(MAX_MOVES):
			# Случайно выбирать из данных трёх ходов
			moves = stockfish.get_top_moves(3)
			# Если доступно меньше трёх ходов, то выбрать первый, если доступных ходов нет, то выйти
			if (len(moves) == 0):
				print("game is over")
				break
			elif (len(moves) == 1):
				move = moves[0]["Move"]
			elif (len(moves) == 2):
				move = random.choices(moves, weights=(80, 20), k=1)[0]["Move"]
			else:
				move = random.choices(moves, weights=(80, 15, 5), k=1)[0]["Move"]

			currentGamePositions.append(stockfish.get_fen_position())
			currentGameMoves.append(move) # Убедиться, что создана стринговая версия хода перед сменой формата
			move = chess.Move.from_uci(str(move)) # Конвертировать в формат chess-package
			board.push(move)
			stockfish.set_position(currentGameMoves)
			if (checkEndCondition(board)):
				print("game is over")
				break
		saveData(currentGameMoves, currentGamePositions)
	

if __name__ == '__main__':
	gamePerCore = 200
	#amountOfGamesArray6C = [gamePerCore, gamePerCore, gamePerCore, gamePerCore, gamePerCore, gamePerCore]
	amountOfGamesArray10C = [gamePerCore, gamePerCore, gamePerCore, gamePerCore, gamePerCore, gamePerCore, gamePerCore, gamePerCore, gamePerCore, gamePerCore]
	start_time = time.time()
	with Pool(10) as p:
		p.map(mineGames, amountOfGamesArray10C)
	end_time = time.time()
	elapsed_time = end_time - start_time
	print('Elapsed time: ', elapsed_time/60)
	print('Amount of games per core: ', gamePerCore)
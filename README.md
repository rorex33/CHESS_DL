# CHESS_DL
 Making a chess bot with deep learning

data - содержит в себе два каталога:
    rawData - для партий, генерируемых в RetrievingDataset.py.
    preparedData - подготовленная для обучения информация о партиях, обработанная в  Encoding.py.

stockfish - ИИ stockfish для игры в шахматы.

Training - содержит в себе файлы для обучения сети и каталоги:
    runs - информация о запусках обучения
    savedModels - сохранённые модели нейронных сетей

    MainTraining.py - содержит основыне функции обучения и сохранения результата.

    Model.py - содержит в себе код модели нейронной сети.

    RunTraining - содержит фукнции запуска обучения и отображения процесса обучения.

    TrainingEncodDecod - содержит функции кодировки и раскадировки датасета.

    PlayGame.py - содержит фукнции для запуска партии с наиболее успешной моделью сети.

RetrievingDataset.py - содержит функции для сбора датасета (stockfish играет сам с собой)

Encoding.py - содержит фукнции для приведения "сырого" датасета к читаемому для нейронной сети виду.


# CHESS TERMS
1) rank - строка, file - столбец.
2) UCI (англ. Universal Chess Interface) — свободно распространяемый коммуникационный протокол, позволяющий движкам шахматных программ взаимодействовать с их графическим интерфейсом.

# TO DO
Добавить Dropout


# Старые значения для нейронных слоёв
#: Определяют линейные слои нейронной сети.
    self.linear1 = torch.nn.Linear(self.INPUT_SIZE, 500)
    self.linear2 = torch.nn.Linear(500, 500)
    self.linear3 = torch.nn.Linear(500, 500)
    self.linear4 = torch.nn.Linear(500, 200)
    self.linear5 = torch.nn.Linear(200, self.OUTPUT_SIZE)

5.326042175292969 (5000 игр)
        #: Определяют линейные слои нейронной сети.
        self.linear1 = torch.nn.Linear(self.INPUT_SIZE, 128)
        self.linear2 = torch.nn.Linear(128, 128)
        self.linear3 = torch.nn.Linear(128, 128)
        self.linear4 = torch.nn.Linear(128, 64)
        self.linear5 = torch.nn.Linear(64, self.OUTPUT_SIZE)

6.540229320526123
        #: Определяют линейные слои нейронной сети.
        self.linear1 = torch.nn.Linear(self.INPUT_SIZE, 512)
        self.linear2 = torch.nn.Linear(512, 512)
        self.linear3 = torch.nn.Linear(512, 512)
        self.linear4 = torch.nn.Linear(512, 128)
        self.linear5 = torch.nn.Linear(128, self.OUTPUT_SIZE)

5.933191299438477
        #: Определяют линейные слои нейронной сети.
        self.linear1 = torch.nn.Linear(self.INPUT_SIZE, 32)
        self.linear2 = torch.nn.Linear(32, 32)
        self.linear3 = torch.nn.Linear(32, 32)
        self.linear4 = torch.nn.Linear(32, 16)
        self.linear5 = torch.nn.Linear(16, self.OUTPUT_SIZE)
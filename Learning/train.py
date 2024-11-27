# %%
import argparse

# %%
from model import CVAE
from utils import *
import numpy as np
import os
import tensorflow as tf
import time
import pandas as pd
import logging
import sys





# %%
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', help='batch_size', type=int, default=128)
parser.add_argument('--latent_size', help='latent_size', type=int, default=200)
parser.add_argument('--unit_size', help='unit_size of rnn cell', type=int, default=512)
parser.add_argument('--n_rnn_layer', help='number of rnn layer', type=int, default=3)
parser.add_argument('--seq_length', help='max_seq_length', type=int, default=120)
parser.add_argument('--prop_file', help='name of property file', type=str)
parser.add_argument('--mean', help='mean of VAE', type=float, default=0.0)
parser.add_argument('--stddev', help='stddev of VAE', type=float, default=1.0)
parser.add_argument('--num_epochs', help='epochs', type=int, default=100)
parser.add_argument('--lr', help='learning rate', type=float, default=0.0001)
parser.add_argument('--num_prop', help='number of propertoes', type=int, default=3)
parser.add_argument('--save_dir', help='save dir', type=str, default='save/')
args = parser.parse_args()


# %%
df_1 = pd.read_csv(r'C:\Users\Igorr\Documents\ITMO5grade\Project_with_Susan\Made_code_github\Made_code\All_csv_files\Final_set.csv', index_col=False)
df_1 = df_1.drop(columns='Unnamed: 0')
# display(df_1)

logging.info(f"DataFrame loaded: shape = {df_1.shape}")
logging.info(f"First rows of DataFrame:\n{df_1.head()}")
# %%
# Имя столбца с SMILES
smiles_col = 'canonical_smiles_x'

# Вычислить длины SMILES
smiles_lengths = df_1[smiles_col].apply(len)

# Найти максимальную длину
max_length = smiles_lengths.max()
# print(f"Максимальная длина SMILES: {max_length}")

# Найти статистику длин
# print(smiles_lengths.describe())

# Определить seq_length с запасом
seq_length = max_length + 2  # Добавляем символы начала и конца
# print(f"Рекомендуемая длина seq_length: {seq_length}")

# %%
def load_data_from_csv(df_1, smiles_col, props_cols, seq_length):
    """
    Загружает данные из DataFrame, преобразует SMILES в входные и выходные массивы.
    
    Args:
        df (pandas.DataFrame): Данные с колонками SMILES и свойств.
        smiles_col (str): Имя колонки с SMILES.
        props_cols (list): Список колонок со свойствами.
        seq_length (int): Максимальная длина последовательности SMILES.
    
    Returns:
        smiles_input (np.array): Массив входных последовательностей.
        smiles_output (np.array): Массив выходных последовательностей.
        chars (tuple): Уникальные символы в словаре.
        vocab (dict): Словарь символов -> индексов.
        prop (np.array): Массив свойств.
        length (np.array): Массив длин SMILES.
    """
    import collections

    # Извлечение SMILES и свойств
    smiles = df_1[smiles_col].tolist()
    props = df_1[props_cols].values  # Свойства как numpy array

    # Фильтрация SMILES, превышающих seq_length
    smiles = [s for s in smiles if len(s) < seq_length - 2]

    # Создание словаря символов
    total_string = ''.join(smiles)
    counter = collections.Counter(total_string)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    chars, counts = zip(*count_pairs)
    vocab = dict(zip(chars, range(len(chars))))

    # Добавление специальных символов
    chars += ('E',)  # End of sequence
    chars += ('X',)  # Start of sequence
    vocab['E'] = len(chars) - 2
    vocab['X'] = len(chars) - 1

    # Длины последовательностей
    length = np.array([len(s) + 1 for s in smiles])

    # Преобразование SMILES в формат для модели
    smiles_input = [('X' + s).ljust(seq_length, 'E') for s in smiles]
    smiles_output = [s.ljust(seq_length, 'E') for s in smiles]
    smiles_input = np.array([np.array(list(map(vocab.get, s))) for s in smiles_input])
    smiles_output = np.array([np.array(list(map(vocab.get, s))) for s in smiles_output])

    return smiles_input, smiles_output, chars, vocab, props, length

# %%
# Параметры
smiles_col = 'canonical_smiles_x'  # Имя столбца с SMILES
# props_cols = ['property1', 'property2', 'property3']  # Столбцы свойств
props_cols = df_1.select_dtypes(include=['number']).columns.tolist()
seq_length = 120  # Максимальная длина последовательности


# %%
# Вызов функции
smiles_input, smiles_output, chars, vocab, props, length = load_data_from_csv(df_1, smiles_col, props_cols, seq_length)
vocab_size = len(vocab)

# %%
num_train_data = int(len(smiles_input) * 0.75)
train_input = smiles_input[:num_train_data]
test_input = smiles_input[num_train_data:]

train_output = smiles_output[:num_train_data]
test_output = smiles_output[num_train_data:]

train_labels = props[:num_train_data]
test_labels = props[num_train_data:]

train_length = length[:num_train_data]
test_length = length[num_train_data:]

# Проверка и создание save_dir
if not os.path.isdir(args.save_dir):
    os.mkdir(args.save_dir)

    def train_step(self, x, y, l, c):
        with tf.GradientTape() as tape:
    # Расчёт потерь
            loss, recon_loss, latent_loss = self((x, y, c, l))
    # Вычисление градиентов
        grads = tape.gradient(loss, self.trainable_variables)
    # Обновление параметров модели
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss.numpy()  # Возвращаем значение потерь

    def test_step(self, x, y, l, c):
        loss, _, _ = self((x, y, c, l))
        return loss.numpy()  # Возвращаем значение потерь

    # Инициализация модели
    model = CVAE(vocab_size, args)

# Цикл обучения
for epoch in range(args.num_epochs):
    train_loss = []
    test_loss = []
    st = time.time()

    # Обучение
    for iteration in range(len(train_input) // args.batch_size):
        n = np.random.randint(len(train_input), size=args.batch_size)
        x = np.array([train_input[i] for i in n])
        y = np.array([train_output[i] for i in n])
        l = np.array([train_length[i] for i in n])
        c = np.array([train_labels[i] for i in n])
        cost = model.train_step(x, y, l, c)
        train_loss.append(cost)

    # Тестирование
    for iteration in range(len(test_input) // args.batch_size):
        n = np.random.randint(len(test_input), size=args.batch_size)
        x = np.array([test_input[i] for i in n])
        y = np.array([test_output[i] for i in n])
        l = np.array([test_length[i] for i in n])
        c = np.array([test_labels[i] for i in n])
        cost = model.test_step(x, y, l, c)
        test_loss.append(cost)

    # Вычисление средней потери
    train_loss = np.mean(train_loss)
    test_loss = np.mean(test_loss)
    end = time.time()

    # Запись метрик после вычисления потерь
    metrics_path = os.path.join(args.save_dir, 'metrics.csv')
    if not os.path.exists(metrics_path):  # Если файл еще не создан, добавьте заголовок
        with open(metrics_path, 'w') as f:
            f.write("epoch,train_loss,test_loss,time\n")

    with open(metrics_path, 'a') as f:
        f.write(f"{epoch + 1},{train_loss:.4f},{test_loss:.4f},{end - st:.2f}\n")

    # Логирование
    print(f"Epoch {epoch + 1}/{args.num_epochs}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, Time = {end - st:.2f}s")

    # Сохранение модели
    ckpt_path = os.path.join(args.save_dir, f'model_{epoch + 1}.ckpt')
    model.save(ckpt_path)
    print(f"Model checkpoint saved at {ckpt_path}")

# # Настройка логирования
#     log_file = os.path.join(args.save_dir, 'script.log')
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s [%(levelname)s]: %(message)s',
#         handlers=[
#             logging.FileHandler(log_file, mode='w'),  # Лог в файл
#             logging.StreamHandler(sys.stdout)  # Лог в консоль
#         ]
#     )

#     logging.info("Начало выполнения скрипта")

#     logging.info(f"Epoch {epoch + 1}/{args.num_epochs}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, Time = {end - st:.2f}s")

# metrics_path = os.path.join(args.save_dir, 'metrics.csv')
# if not os.path.exists(metrics_path):
#     with open(metrics_path, 'w') as f:
#         f.write("epoch,train_loss,test_loss,time\n")

# # Внутри цикла обучения:
# with open(metrics_path, 'a') as f:
#     f.write(f"{epoch + 1},{train_loss:.4f},{test_loss:.4f},{end - st:.2f}\n")
# logging.info(f"Metrics saved for epoch {epoch + 1} to {metrics_path}")

# ckpt_path = os.path.join(args.save_dir, f'model_{epoch}.ckpt')
# model.save(ckpt_path)
# logging.info(f"Model checkpoint saved at {ckpt_path}")

# predictions = model(test_input)  # Получить предсказания
# predictions_path = os.path.join(args.save_dir, 'predictions.csv')
# np.savetxt(predictions_path, predictions, delimiter=',')
# logging.info(f"Predictions saved to {predictions_path}")
# try:
#     # Ваш код
#     # Например, цикл обучения:
#     for epoch in range(args.num_epochs):
#     # Обучение и логирование
#         pass
# except Exception as e:
#         logging.error(f"Ошибка во время выполнения: {e}", exc_info=True)
#         sys.exit(1)
# import logging
# import os
# import numpy as np
# import time

# # Настройка логирования
# log_file = 'save/script.log'
# if not os.path.exists('save'):
#     os.makedirs('save')
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s [%(levelname)s]: %(message)s',
#     handlers=[
#         logging.FileHandler(log_file, mode='w'),
#         logging.StreamHandler()
#     ]
# )

# logging.info("Начало выполнения скрипта")

# # Параметры
# num_epochs = 10  # Задайте количество эпох для теста
# batch_size = 128
# save_dir = 'save/'

# # Инициализация данных (заглушка)
# train_loss = []
# test_loss = []

# # Заглушка для модели (пример для теста)
# class DummyModel:
#     def train_step(self, x, y, l, c):
#         return np.random.random()

#     def test_step(self, x, y, l, c):
#         return np.random.random()

#     def save(self, path):
#         logging.info(f"Модель сохранена в {path}")


# model = DummyModel()

# # Цикл обучения
# for epoch in range(num_epochs):
#     epoch_train_loss = []
#     epoch_test_loss = []
#     st = time.time()

#     # Обучение
#     for _ in range(5):  # Количество шагов в эпохе
#         cost = model.train_step(None, None, None, None)  # Заглушка
#         epoch_train_loss.append(cost)

#     # Тестирование
#     for _ in range(5):  # Количество шагов в эпохе
#         cost = model.test_step(None, None, None, None)  # Заглушка
#         epoch_test_loss.append(cost)

#     # Средние потери за эпоху
#     train_loss = np.mean(epoch_train_loss)
#     test_loss = np.mean(epoch_test_loss)
#     end = time.time()

#     # Сохранение метрик
#     metrics_path = os.path.join(save_dir, 'metrics.csv')
#     if not os.path.exists(metrics_path):
#         with open(metrics_path, 'w') as f:
#             f.write("epoch,train_loss,test_loss,time\n")
#     with open(metrics_path, 'a') as f:
#         f.write(f"{epoch + 1},{train_loss:.4f},{test_loss:.4f},{end - st:.2f}\n")

#     logging.info(f"Epoch {epoch + 1}/{num_epochs}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, Time = {end - st:.2f}s")

#     # Сохранение модели
#     ckpt_path = os.path.join(save_dir, f'model_{epoch + 1}.ckpt')
#     model.save(ckpt_path)

# logging.info("Обучение завершено")


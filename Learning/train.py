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

# print (args)
# #convert smiles to numpy array
# molecules_input, molecules_output, char, vocab, labels, length = load_data(args.prop_file, args.seq_length)
# vocab_size = len(char)

# #make save_dir
# if not os.path.isdir(args.save_dir):
#     os.mkdir(args.save_dir)

# #divide data into training and test set
# num_train_data = int(len(molecules_input)*0.75)
# train_molecules_input = molecules_input[0:num_train_data]
# test_molecules_input = molecules_input[num_train_data:-1]

# train_molecules_output = molecules_output[0:num_train_data]
# test_molecules_output = molecules_output[num_train_data:-1]

# train_labels = labels[0:num_train_data]
# test_labels = labels[num_train_data:-1]

# train_length = length[0:num_train_data]
# test_length = length[num_train_data:-1]

# model = CVAE(vocab_size,
#             args
#             )
# print ('Number of parameters : ', np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))

# for epoch in range(args.num_epochs):

#     st = time.time()
#     # Learning rate scheduling 
#     #model.assign_lr(learning_rate * (decay_rate ** epoch))
#     train_loss = []
#     test_loss = []
#     st = time.time()
    
#     for iteration in range(len(train_molecules_input)//args.batch_size):
#         n = np.random.randint(len(train_molecules_input), size = args.batch_size)
#         x = np.array([train_molecules_input[i] for i in n])
#         y = np.array([train_molecules_output[i] for i in n])
#         l = np.array([train_length[i] for i in n])
#         c = np.array([train_labels[i] for i in n])
#         cost = model.train(x, y, l, c)
#         train_loss.append(cost)
    
#     for iteration in range(len(test_molecules_input)//args.batch_size):
#         n = np.random.randint(len(test_molecules_input), size = args.batch_size)
#         x = np.array([test_molecules_input[i] for i in n])
#         y = np.array([test_molecules_output[i] for i in n])
#         l = np.array([test_length[i] for i in n])
#         c = np.array([test_labels[i] for i in n])
#         cost = model.test(x, y, l, c)
#         test_loss.append(cost)
    
#     train_loss = np.mean(np.array(train_loss))        
#     test_loss = np.mean(np.array(test_loss))    
#     end = time.time()    
#     if epoch==0:
#         print ('epoch\ttrain_loss\ttest_loss\ttime (s)')
#     print ("%s\t%.3f\t%.3f\t%.3f" %(epoch, train_loss, test_loss, end-st))
#     ckpt_path = args.save_dir+'/model_'+str(epoch)+'.ckpt'
#     model.save(ckpt_path, epoch)

# %%
df_1 = pd.read_csv(r'C:\Users\Igorr\Documents\ITMO5grade\Project_with_Susan\Made_code_github\Made_code\All_csv_files\Final_set.csv', index_col=False)
df_1 = df_1.drop(columns='Unnamed: 0')
# display(df_1)

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

# %%
# # Преобразование данных
# molecules_input, molecules_output, chars, vocab, labels, length = load_data_from_csv(
#     df_1, smiles_col=smiles_col, props_cols=props_cols, seq_length=seq_length
# )



# # Разделение данных
# train_input, test_input, train_output, test_output, train_labels, test_labels, train_length, test_length = train_test_split(
#     molecules_input, molecules_output, labels, length, test_size=0.25, random_state=42
# )

# Проверка и создание save_dir
if not os.path.isdir(args.save_dir):
    os.mkdir(args.save_dir)

# Инициализация модели
model = CVAE(vocab_size, args)

# print('Number of parameters:', np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))

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
        cost = model.train(x, y, l, c)
        train_loss.append(cost)
    
    # Тестирование
    for iteration in range(len(test_input) // args.batch_size):
        n = np.random.randint(len(test_input), size=args.batch_size)
        x = np.array([test_input[i] for i in n])
        y = np.array([test_output[i] for i in n])
        l = np.array([test_length[i] for i in n])
        c = np.array([test_labels[i] for i in n])
        cost = model.test(x, y, l, c)
        test_loss.append(cost)
    
    train_loss = np.mean(train_loss)
    test_loss = np.mean(test_loss)
    end = time.time()
    
    # if epoch == 0:
        # print('epoch\ttrain_loss\ttest_loss\ttime (s)')
    # print(f"{epoch}\t{train_loss:.3f}\t{test_loss:.3f}\t{end - st:.3f}")
    
    # Сохранение модели
    ckpt_path = os.path.join(args.save_dir, f'model_{epoch}.ckpt')
    model.save(ckpt_path, epoch)

# %%
# import json
# import os
# import numpy as np
# import tensorflow as tf
# import time

# # Загрузка конфигурации из JSON-файла
# def load_config(config_path):
#     with open(config_path, 'r') as file:
#         config = json.load(file)
#     return config

# # Путь к файлу конфигурации
# config_path = r'C:\Users\Igorr\Documents\ITMO5grade\Project_with_Susan\Made_code_github\Made_code\Learning\config_train.json'

# # Загружаем параметры
# config = load_config(config_path)

# # Распаковываем параметры из конфигурации
# batch_size = config.get('batch_size', 128)
# latent_size = config.get('latent_size', 200)
# unit_size = config.get('unit_size', 512)
# n_rnn_layer = config.get('n_rnn_layer', 3)
# seq_length = config.get('seq_length', 120)
# prop_file = config.get('prop_file', None)
# mean = config.get('mean', 0.0)
# stddev = config.get('stddev', 1.0)
# num_epochs = config.get('num_epochs', 100)
# lr = config.get('lr', 0.0001)
# num_prop = config.get('num_prop', 3)
# save_dir = config.get('save_dir', 'save/')

# # #convert smiles to numpy array
# # molecules_input, molecules_output, char, vocab, labels, length = load_data(prop_file, seq_length)
# # vocab_size = len(char)



# %%
# print(molecules_input)
# print(molecules_output)
# print(char)
# print(vocab)

# %%
# def load_data_from_dataframe(df_1, seq_length):
#     # Извлекаем SMILES строки
#     molecules = df_1['canonical_smiles_x'].tolist()

#     # Извлекаем свойства (все колонки, кроме SMILES, используются как свойства)
#     labels = df_1.drop(columns=['canonical_smiles_x']).values

#     # Извлекаем уникальные символы
#     char = sorted(set(''.join(molecules)))
#     vocab = {c: i for i, c in enumerate(char)}

#     # Преобразование SMILES в числовые последовательности
#     molecules_input = []
#     molecules_output = []

#     for smile in molecules:
#         # Преобразуем SMILES в числовые индексы
#         smile_indices = [vocab[char] for char in smile]

#         # Дополняем последовательности до seq_length
#         if len(smile_indices) < seq_length:
#             smile_indices += [vocab[char[-1]]] * (seq_length - len(smile_indices))  # Padding

#         # Разделяем input и output
#         molecules_input.append(smile_indices[:-1])  # Без последнего символа
#         molecules_output.append(smile_indices[1:])  # Без первого символа

#     # Проверка длины всех последовательностей
#     molecules_input = [seq[:seq_length - 1] + [vocab[char[-1]]] * (seq_length - len(seq)) for seq in molecules_input]
#     molecules_output = [seq[:seq_length - 1] + [vocab[char[-1]]] * (seq_length - len(seq)) for seq in molecules_output]

#     # Преобразуем в numpy массивы
#     molecules_input = np.array(molecules_input)
#     molecules_output = np.array(molecules_output)

#     # Длины последовательностей
#     length = [min(len(smile), seq_length) for smile in molecules]

#     return molecules_input, molecules_output, char, vocab, labels, length





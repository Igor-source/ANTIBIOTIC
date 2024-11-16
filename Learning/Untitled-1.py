# %%
import joblib

# Загрузка файла .jbl
filename = r"C:\Users\Igorr\Documents\ITMO5grade\Project_with_Susan\Made_code_github\Made_code\Learning\dataset.jbl"
data = joblib.load(filename)

# Просмотр содержимого
print(data)

# %%
import pandas as pd
import numpy as np
import csv
import os, sys
import json
import joblib
import argparse
from rdkit import Chem
import openbabel
import pybel
from rdkit.Chem import rdMolDescriptors

# %%
import oddt.toolkits.extras.rdkit as ordkit

# %%
df_1 = pd.read_csv(r'C:\Users\Igorr\Documents\ITMO5grade\Project_with_Susan\Made_code_github\Made_code\All_csv_files\Final_set.csv', index_col=False)
df_1 = df_1.drop(columns='Unnamed: 0')
display(df_1)

# %%
molname_list = df_1[['molecule_chembl_id_x','canonical_smiles_x']]
molname_list = molname_list.drop_duplicates()
# Сброс индексов
molname_list = molname_list.reset_index(drop=True)
display(molname_list)

# %%
import numpy as np
import tensorflow as tf
import numpy as np
import threading


# %%
df_smiles = molname_list['canonical_smiles_x'].tolist()
df_smiles


# %%
with open('smiles.txt', 'w') as file:
    for item in df_smiles:
        file.write(f"{item}\n")

# %%
# import argparse
# from rdkit.Chem import MolFromSmiles
# from rdkit.Chem.Descriptors import ExactMolWt, MolLogP
# from rdkit.Chem.rdMolDescriptors import CalcTPSA
# from multiprocessing import Pool

# %%
from tqdm import tqdm


# %%
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Descriptors import ExactMolWt, MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from multiprocessing import Pool
from tqdm import tqdm  # Импортируем tqdm для прогресс-бара

# Задаем значения для переменных
input_filename = r'C:\Users\Igorr\Documents\ITMO5grade\Project_with_Susan\Made_code_github\Made_code\Data_curation\smiles.txt'
output_filename = r'C:\Users\Igorr\Documents\ITMO5grade\Project_with_Susan\Made_code_github\Made_code\Data_curation\smiles_prop.txt'
ncpus = 1  # количество процессоров для параллельной обработки

# def cal_prop(s):
#     m = MolFromSmiles(s)
#     if m is None: 
#         return None
#     return MolFromSmiles(s), ExactMolWt(m), MolLogP(m), CalcTPSA(m)
def cal_prop(s):
    print(f"Processing SMILES: {s}")
    try:
        m = MolFromSmiles(s)
        if m is None: 
            return None
        return MolFromSmiles(s), ExactMolWt(m), MolLogP(m), CalcTPSA(m)
    except Exception as e:
        print(f"Error processing SMILES '{s}': {e}")
        return None

# Читаем SMILES и используем tqdm для отображения прогресса
with open(input_filename) as f:
    smiles = f.read().splitlines()
    
# # Оборачиваем pool.map с tqdm для отслеживания прогресса
# with Pool(ncpus) as pool:
#     data = list(tqdm(pool.imap(cal_prop, smiles), total=len(smiles)))
with Pool(ncpus) as pool:
    data = list(tqdm(pool.imap(cal_prop, smiles[:10]), total=10))

# Записываем результаты в выходной файл
with open(output_filename, 'w') as w:
    for d in data:
        if d is None:
            continue
        w.write(f"{d[0]}\t{d[1]}\t{d[2]}\t{d[3]}\n")

# %%
result = cal_prop(smiles[0])  # Первую строку SMILES обрабатываем отдельно
display(result)

# %%
data = []
for s in tqdm(smiles):
    data.append(cal_prop(s))

# %%




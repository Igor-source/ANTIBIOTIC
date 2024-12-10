## Project Structure
```
project_root/
│
├── Split_Mordred_set/       # Split into multiple CSV files due to large size. Contains all descriptors from the Mordred library.
├── Split_data_Merged_data/  # Split into multiple CSV files due to large size. This is the merged dataset of Mordred, RdKit, and new_qm9 descriptors.
├── .gitignore               # Specifies intentionally untracked files to ignore.
├── Merged_data2.csv         # A merged dataset in CSV format.
├── NewDataset.csv           # A new dataset generated for analysis.
├── Project1.ipynb           # Extracting descriptors from Mordred and RdKit.
├── Project2.ipynb           # Feature selection methods: Pearson correlation and visualization.
├── Project3.ipynb           # Data curation: encoding categorical data, detecting outliers.
├── Project4 (1).ipynb       # Feature transformation methods: PCA and t-SNE.
├── Project5.ipynb           # Demonstration of additional data processing methods using t-SNE.
├── Project6.ipynb           # Model training with XGBoost and LightGBM.
├── README.md                # Project documentation.
├── RdKitSet.csv             # Dataset containing RdKit descriptors.
├── Without_HOMO_LUMO.csv    # Final dataset prepared for PostgreSQL database upload.
├── new_qm9.csv              # The original dataset used initially.
├── split_csv.ipynb          # Script to split large datasets into smaller parts.
└── transformed_df.csv       # Dataset after normalization and dimensionality reduction.
```

Data_curation №
1Data_collection.ipynb  № Сбор антибиотиков из библиотеки Chembl. По итогам оценки датасета, в нем данные не для всех типов концентраций. Проводим извлечение всех типов для 95 уникальных молекул 
Activity_drugs.ipynb  № В этом коде удаляем типы концентраций, встречающихся меньше 100 раз 
Descriptors_drugs.ipynb  № Получение дескрипторов антибиотиков из библиотеки RdKit
Diagramandgraph.ipynb  # Визуализация units и type. Вконце собирается датасет для определения таксонов
Merge.ipynb   # Объединение данных концентрации с дескрипторами в начале и под конец уже и с таксономией. В итоге получаем final dataset
Split_bigcsv_file.ipynb  # Файл позволяющий разбить большой датасет на несколько для выгрузки в гитхаб
Taxonomy.ipynb   № Определяется таксономия и отбираются только бактерии
Units_curation.ipynb   № Все размерности приводятся к одному виду µg/ml
jupyter.ipynb
requirements.txt
smiles.txt
smiles_prop.txt

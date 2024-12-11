## Project Structure
```
project_root/
│
├── Data_curation/           # Scripts and notebooks for data cleaning and preparation.
│   ├── Data_collection.ipynb  # Collection of antibiotic data from the Chembl library. Extracts all concentration types for 95 unique molecules.
│   ├── Activity_drugs.ipynb   # Filters out concentration types appearing less than 100 times.
│   ├── Descriptors_drugs.ipynb # Generates antibiotic descriptors using the RdKit library.
│   ├── Diagramandgraph.ipynb  # Visualizes units and types; finalizes a dataset for taxonomic analysis.
│   ├── Merge.ipynb            # Merges concentration data with descriptors and taxonomy, resulting in the final dataset.
│   ├── Split_bigcsv_file.ipynb # Splits a large dataset into smaller files for GitHub upload.
│   ├── Taxonomy.ipynb         # Identifies taxonomy and selects only bacterial data.
│   ├── Units_curation.ipynb   # Standardizes units to µg/ml.
│   └── jupyter.ipynb          # Miscellaneous notebook for testing and temporary analysis.
│
├── smiles.txt               # SMILES representations of molecules.
├── smiles_prop.txt          # Molecule properties associated with SMILES.
├── requirements.txt         # Python dependencies for the project.
└── README.md                # Project documentation.
```


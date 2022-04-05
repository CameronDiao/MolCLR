import os
import csv
import math
import time
import random
import copy


import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger  

smiles_data = []
mols_data = []

with open("../data/bbbp/BBBP.csv") as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=',')
    for i, row in enumerate(csv_reader):
        if i != 0:
            smiles = row['smiles']
            label = row['p_np']
            name = row['name']
            mol = Chem.MolFromSmiles(smiles)
            if mol != None and label != '':
                smiles_data.append(smiles)
                mols_data.append(name)

with open("../data/bbbp/BBBP.txt", 'w') as f:
    for i in range(len(smiles_data)):
        f.write("%(smiles)s\t%(name)s\n" % {'smiles': smiles_data[i], 'name': mols_data[i]})


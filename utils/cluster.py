import copy
import time
import random
import pathlib as Path

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.Chem import Draw
from rdkit.Chem import rdFingerprintGenerator

import jarvispatrick

def tanimoto_dist_matrix(finger_data):
    dist_matrix = []
    for i in range(1, len(finger_data)):
        # compare current fingerprint against all previous ones in the fingerprints list
        sims = DataStructs.BulkTanimotoSimilarity(finger_data[i], finger_data[:i])
        # (we need a dist matrix) so calculate (1 - element) for every element in the sim matrix
        dist_matrix.extend([1 - s for s in sims])
    return dist_matrix

def butina_clustering(smiles_data, cutoff=0.2):
    mol_data = [Chem.MolFromSmiles(smiles) for smiles in smiles_data]
    finger_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5)
    finger_data = [finger_gen.GetFingerprint(mol) for mol in mol_data]

    dist_matrix = tanimoto_dist_matrix(finger_data)
    but_clusters = Butina.ClusterData(dist_matrix, len(finger_data), cutoff, isDistData=True)
    but_clusters = sorted(but_clusters, key=len, reverse=True)
    #but_centroids = [mol_data[c[0]] for c in but_clusters]
    #sorted_clusters = []
    #for cluster in but_clusters:
    #    if len(clusters) <= 1:
    #        continue
    #    sorted_finger = [finger_gen.GetFingerprint(mol_data[i]) for i in cluster]
    #    sims = DataStructs.BulkTanimotoSimilarity(sorted_finger[0], sorted_finger[1:])
    #    sims = list(zip(sims, cluster[1:]))
    #    sims.sort(reverse=True)
    #    sorted_clusters.append((len(sims), [i for __, i in sims]))
    #    sorted_clusters.sort(reverse=True)
    #sel_mols = but_centroids.copy()
    #idx = 0
    #pending = 1000 - len(sel_mols)
    #while pending > 0 and idx < len(sorted_clusters):
    #    tmp_cluster = sorted_clusters[idx][1]
    #    if sorted_clusters[idx][0] > 10:
    #        num_mols = 10
    #    else:
    #        num_mols = int(0.5 * len(tmp_cluster)) + 1
    #    if num_mols > pending:
    #        num_mols = pending
    #    sel_mols += [mol_data[i] for i in tmp_cluster[:num_mols]]
    #    idx += 1
    #    pending = 1000 - len(sel_mols)
        
    return but_clusters

def jp_clustering(smiles_data, ex_neigh=10, comm_neigh=2):
    def compute_tan_sim(m1, m2):
        fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, 2)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, 2)
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    mol_data = [Chem.MolFromSmiles(smiles) for smiles in smiles_data]
    cluster_gen = jarvispatrick.JarvisPatrick(mol_data, compute_tan_sim)
    jp_clusters = cluster_gen(ex_neigh, comm_neigh)
    return jp_clusters


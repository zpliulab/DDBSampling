import torch.utils.data
import random, os, tqdm
import numpy as np
from Bio import SeqIO
from Bio.SeqUtils import seq1, seq3
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch_geometric.data import Data
import networkx as nx
Amino_acid_type = [
"ILE",
"VAL",
"LEU",
"PHE",
"CYS",
"MET",
"ALA",
"GLY",
"THR",
"SER",
"TRP",
"TYR",
"PRO",
"HIS",
"GLU",
"GLN",
"ASP",
"ASN",
"LYS",
"ARG",
]
Amino_acid_dict = {item:index+1 for index, item in enumerate(Amino_acid_type)} #1~20, others 21, padding 0.
base_type = ['A', 'C', 'G', 'T']
Base_dict = {item:index+1 for index, item in enumerate(base_type)}

def pack(mol1=None,
         mol2=None,
         token_1=None,
         token_2=None,
         label=None):
    pair = {}
    pair['token_1'] = token_1
    pair['token_2'] = token_2
    pair['mol_1'] = mol1
    pair['mol_2'] = mol2
    pair['label'] = label
    return pair

num_atom_feat = 34
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom,explicit_H=False,use_chirality=True):
    """Generate atom features including atom symbol(10),degree(7),formal charge,
    radical electrons,hybridization(6),aromatic(1),Chirality(3)
    """
    symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']  # 10-dim
    degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
    hybridizationType = [Chem.rdchem.HybridizationType.SP,
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3,
                              Chem.rdchem.HybridizationType.SP3D,
                              Chem.rdchem.HybridizationType.SP3D2,
                              'other']   # 6-dim
    results = one_of_k_encoding_unk(atom.GetSymbol(),symbol) + \
                  one_of_k_encoding(atom.GetDegree(),degree) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [atom.GetIsAromatic()]  # 10+7+2+6+1=26

    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])   # 26+5=31
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 31+3 =34
    return results


def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency,dtype=np.float32)


def mol_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        raise RuntimeError("SMILES cannot been parsed!")
    #mol = Chem.AddHs(mol)
    atom_feat = np.zeros((mol.GetNumAtoms(), num_atom_feat))
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    # adj_matrix = adjacent_matrix(mol)
    return torch.FloatTensor(atom_feat), torch.LongTensor(edge_index).T

class PairLoader():
    """multi-threaded data loading"""

    def __init__(self, args, sample_dir=None):
        self.args = args
        self.inter_list, self.pro_list = self.obtain_ppi_list(sample_dir)
        self.data_dir = 'dataset/{}'.format(args.dataset_name)
        if args.task == 'PPI':
            self.fea_1_dict = self.load_protein_fasta(mol=1)
            self.fea_2_dict = self.fea_1_dict
        elif args.task == 'RPI':
            self.fea_1_dict = self.load_RNA_fasta(mol=1)
            self.fea_2_dict = self.load_protein_fasta(mol=2)
        elif args.task == 'DTI':
            self.fea_1_dict = self.load_Grug_fasta(mol=1)
            self.fea_2_dict = self.load_protein_fasta(mol=2)

    def load_protein_fasta(self, mol=1):
        fasta_dir = os.path.join(self.data_dir, 'fasta{}.txt'.format(mol))
        fasta_list = SeqIO.parse(fasta_dir, "fasta")
        seq_features = {}
        for seq_record in tqdm.tqdm(fasta_list):
            feature = torch.LongTensor([Amino_acid_dict.get(seq3(token).upper(), len(Amino_acid_dict)+1) for token in seq_record.seq])
            name = seq_record.id if 'sp|' not in seq_record.id else seq_record.id[3:9]
            seq_features[name] = feature
        return seq_features

    def load_RNA_fasta(self, mol=1):
        fasta_dir = os.path.join(self.data_dir, 'fasta{}.txt'.format(mol))
        fasta_list = SeqIO.parse(fasta_dir, "fasta")
        seq_features = {}
        for seq_record in tqdm.tqdm(fasta_list):
            feature = torch.LongTensor([Base_dict.get((token).upper(), len(Base_dict)+1) for token in seq_record.seq])
            seq_features[seq_record.id] = feature
        return seq_features

    def load_Grug_fasta(self, mol=1):

        fasta_dir = os.path.join(self.data_dir, 'drugs.smi'.format(mol))
        with open(fasta_dir, 'r') as pid:
            lines = pid.readlines()

        drug_graph = {}
        for seq_record in tqdm.tqdm(lines):
            smile, name = seq_record.strip().split()
            try:
                atom_feat, adj_index = mol_features(smile)
                assert len(atom_feat) > 1 and len(adj_index) > 0
                drug_graph[name] = Data(x=atom_feat, edge_index=adj_index)
            except:
                pass
        
        return drug_graph

    def obtain_ppi_list(self, sample_dir):
        # if os.path.exists(os.path.join(data_dir, 'train_random.txt')):
        #     return
        with open(sample_dir, 'r') as pid:
            lines=pid.readlines()
        inter = []
        pro = []
        for line in lines:
            p1_p2_label = line.strip().split()
            inter.append(p1_p2_label)
            pro.extend(p1_p2_label[:-1])
        pro = list(set(pro))
        return inter, pro

    def __len__(self):
        return len(self.inter_list)

    def __iter__(self):
        for i, p1_p2_label in enumerate(self.inter_list):
            if i >= len(self.inter_list):
                break
            p1, p2, label = self.inter_list[i]
            if p1 not in self.fea_1_dict.keys() or p2 not in self.fea_2_dict.keys():
                # print('hello')
                continue

            p1_torch = self.fea_1_dict[p1]
            p2_torch = self.fea_2_dict[p2]

            yield pack(p1, p2 , p1_torch, p2_torch, torch.LongTensor([int(label),]))

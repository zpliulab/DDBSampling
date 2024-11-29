from subprocess import Popen, PIPE
import numpy as np
from Bio import SeqIO
import os
import pickle
from tqdm import tqdm
from source.contrast_learning.encoder import PPI_encoder, RPI_encoder
import torch

class RNA_Init_Fea:
    def __init__(self, args):
        self.fea_sort = self.__get_fea_sort()
    def __get_fea_sort(self):
        groups = ['A', 'C', 'G', 'T']
        fea_sort = []
        for i in groups:
            for j in groups:
                for m in groups:
                    for n in groups:
                        fea_sort.append(i + j + m + n)
        return fea_sort

    def get_feature(self, seq):
        seq = seq.seq
        q4_to_one = []
        seq = list(str(seq).upper())
        for i in range(len(seq) - 3):
            q4_to_one.append(seq[i] + seq[i + 1] + seq[i + 2] + seq[i + 3])
        feature = []
        for item in self.fea_sort:
            num = q4_to_one.count(item)
            feature.append(float(num) / len(seq))
        return feature

class Protein_Init_Fea:
    def __init__(self, args):
        self.mapping7_groups = self.__get_7groups_mapping()
        self.fea_sort = self.__get_fea_sort()
        pass

    def __get_7groups_mapping(self):
        groups = ['AGV', 'ILFP', 'YMTS', 'HNQW', 'RK', 'DE', 'C']
        groups_list = ['0', '1', '2', '3', '4', '5', '6']
        mapping7_groups = {}
        index = 0
        for group in groups:
            g_members = sorted(group)
            for c in g_members:
                mapping7_groups[c] = str(groups_list[index])
            index = index + 1
        return mapping7_groups

    def __get_fea_sort(self):
        fea_sort = []
        for i in range(7):
            i = str(i)
            for j in range(7):
                j = str(j)
                for m in range(7):
                    m = str(m)
                    fea_sort.append(i + j + m)
        return fea_sort

    def get_feature(self, seq):
        seq = seq.seq
        seq_to_int = []
        for item in seq:
            if item in 'UX' :
                continue
            else:
                seq_to_int.append(self.mapping7_groups[item])
        q3_to_one = []
        for i in range(len(seq_to_int) - 2):
            q3_to_one.append(seq_to_int[i] + seq_to_int[i + 1] + seq_to_int[i + 2])
        feature = []
        for item in self.fea_sort:
            num = q3_to_one.count(item)
            feature.append(float(num) / len(seq_to_int))
        return feature

class Drug_Init_Fea:
    def __init__(self, args):
        Drug_encoder = 'dataset/{}'.format(args.dataset_name)

        with open(os.path.join(Drug_encoder, 'fasta1.txt'), 'r') as pid:
            lines = pid.readlines()
        self.map4_feature_dict = {}
        for line in lines:
            pound_name = line.split('\t')[0].split(' ')[1]
            feature = line.strip().split('\t')[2].split(';')
            self.map4_feature_dict[pound_name] = np.array(feature, dtype=np.float)

    def get_feature(self, seq):
        return self.map4_feature_dict[seq.id]

class mol1fea_mol2fea:
    def __init__(self, args):
        self.args = args
        self.alpha = 0.2

        if args.task == "RPI":
            self.m1_fea_encoder = RNA_Init_Fea
            self.m2_fea_encoder = Protein_Init_Fea
        if args.task == "PPI":
            self.m1_fea_encoder = Protein_Init_Fea
            self.m2_fea_encoder = Protein_Init_Fea
        if args.task == "DTI":
            self.m1_fea_encoder = Drug_Init_Fea
            self.m2_fea_encoder = Protein_Init_Fea

        self.data_dir = 'dataset/{}'.format(args.dataset_name)

    def get_features_seq(self, mol=1):
        if mol == 1:
            fea_encoder = self.m1_fea_encoder(self.args)

        else:
            fea_encoder = self.m2_fea_encoder(self.args)

        if os.path.exists(os.path.join(self.data_dir, 'mol_feature_{}'.format(mol))):
            with open(os.path.join(self.data_dir, 'mol_feature_{}'.format(mol)), 'rb') as pid:
                fea_dict = pickle.load(pid)
            return fea_dict

        print('Extracting mol{} feature'.format(mol))
        class seq_represent:
            def __init__(self, id, seq):
                self.id = id
                self.seq = seq
        if self.args.task == 'DTI' and mol == 1:
            fasta_dir = os.path.join(self.data_dir, 'fasta{}.txt'.format(mol))
            with open(fasta_dir, 'r') as pid:
                lines = pid.readlines()
                fasta_list = []
                for line in lines:
                    id, seq = line.strip().split('\t')[0].split(' ')[1], line.strip().split('\t')[2]
                    fasta_list.append(seq_represent(id, seq))
            pass
        else:
            fasta_dir = os.path.join(self.data_dir, 'fasta{}.txt'.format(mol))
            fasta_list = SeqIO.parse(fasta_dir, "fasta")
            fasta_list = [seq_represent(item.id, item.seq) for item in fasta_list]

        seq_features = []
        name = []
        for seq_record in tqdm(fasta_list):
            feature = fea_encoder.get_feature(seq_record)
            seq_features.append(list(feature))
            name.append(seq_record.id if 'sp|' not in seq_record.id else seq_record.id[3:9])
        seq_features = np.array(seq_features)
        from sklearn.preprocessing import StandardScaler
        seq_features = StandardScaler().fit_transform(seq_features)
        fea_dict = {name[i]: seq_features[i] for i in range(len(name))}
        with open(os.path.join(self.data_dir, 'mol_feature_{}'.format(mol)), 'wb') as pid:
            pickle.dump(fea_dict, pid)
        return fea_dict

    def get_features_contrast(self, mol=1):
        fea_dict = self.get_features_seq(mol)
        if self.args.task =='PPI':
            clf = PPI_encoder()
        else:
            clf = RPI_encoder()
        weight = torch.load('source/contrast_learning/{}_model.ckpt'.format(self.args.dataset_name))
        clf.load_state_dict(weight['state_dict'])
        new_fea_dict = {}
        for name in fea_dict.keys():
            seq_fea = fea_dict[name]
            new_fea = clf.embedding(torch.from_numpy(seq_fea[None,:]).float(), mol=mol)[0,:]
            new_fea_dict[name] = new_fea.detach().numpy()
        return new_fea_dict

    def get_features_noise(self, mol=1):
        if os.path.exists(os.path.join(self.data_dir, 'mol_noise_feature_{}'.format(mol))):
            with open(os.path.join(self.data_dir, 'mol_noise_feature_{}'.format(mol)), 'rb') as pid:
                fea_dict = pickle.load(pid)
            return fea_dict
        class seq_represent:
            def __init__(self, id, seq):
                self.id = id
                self.seq = seq

        if self.args.task == 'DTI' and mol == 1:
            fasta_dir = os.path.join(self.data_dir, 'fasta{}.txt'.format(mol))
            with open(fasta_dir, 'r') as pid:
                lines = pid.readlines()
                fasta_list = []
                for line in lines:
                    id, seq = line.strip().split('\t')[0].split(' ')[1], line.strip().split('\t')[2]
                    fasta_list.append(seq_represent(id, seq))
            pass
        else:
            fasta_dir = os.path.join(self.data_dir, 'fasta{}.txt'.format(mol))
            fasta_list = SeqIO.parse(fasta_dir, "fasta")
        if self.args.task == 'DTI' and mol == 1:
            dimension_size = 128
        elif self.args.task =='RPI' and mol == 1:
            dimension_size = 256
        else:
            dimension_size = 343
        fea_dict = {}
        for item in fasta_list:
            id = item.id if 'sp|' not in item.id else item.id[3:9]
            fea_dict[id] = np.random.normal(size=(dimension_size))
        with open(os.path.join(self.data_dir, 'mol_noise_feature_{}'.format(mol)), 'wb') as pid:
            pickle.dump(fea_dict, pid)
        return fea_dict

    def get_features(self, mol=1):
        if self.args.classifier == 'NOISE_RF':
            # return self.get_features_contrast(mol)
            return self.get_features_noise(mol)
        else:
            return self.get_features_seq(mol)
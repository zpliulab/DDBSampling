from Arguments import parser
import os
from source.exteact_seq_feas import RNA_Init_Fea, Protein_Init_Fea, mol1fea_mol2fea
from prepare_dataset import make_dataset_c1v2c3, make_dataset_CV
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from source.nn_model.model import AE
from source.data_loader import PairLoader
import torch
import pandas as pd
import torch.nn.functional as F
def SEQ_RF():
    clf = RandomForestClassifier(n_estimators=500, criterion='entropy', random_state=0)
    return clf

def train_c1c2c3(args):
    make_dataset_c1v2c3(args)
    data_dir = 'dataset/{}'.format(args.dataset_name)

    if args.neg_sampling_strategy == 'RANDOM':
        sample_dir = os.path.join(data_dir, 'c1_random.txt')
    if args.neg_sampling_strategy == 'RANDOM_GO':
        sample_dir = os.path.join(data_dir, f'c1_random_GO_{args.GO_sim}.txt')
    if args.neg_sampling_strategy == 'RANDOM_distance':
        sample_dir = os.path.join(data_dir, f'c1_random_distance_{args.distance}.txt')
    if args.neg_sampling_strategy == 'RANDOM_subcellular':
        sample_dir = os.path.join(data_dir, f'c1_random_subcellular.txt')
    if args.neg_sampling_strategy == 'DDB':
        sample_dir = os.path.join(data_dir, 'c1_DDB.txt')
    if args.neg_sampling_strategy == 'DDB_GO':
        sample_dir = os.path.join(data_dir, f'c1_DDB_GO_{args.GO_sim}.txt')
    if args.neg_sampling_strategy == 'DDB_subcellular':
        sample_dir = os.path.join(data_dir, 'c1_DDB_subcellular.txt')
    if args.neg_sampling_strategy == 'DDB_distance':
        sample_dir = os.path.join(data_dir, f'c1_DDB_distance_{args.distance}.txt')
        # val_dir = os.path.join(data_dir, 'c1_random.txt')
    train_pair_loader = PairLoader(args, sample_dir)
    train_dataset = [data for data in train_pair_loader]
    model = AE(args).to(args.device)
    model.fit(train_dataset)

def train_cv(args):
    model = AE(args).to(args.device)

    make_dataset_CV(args)
    data_dir = 'dataset/{}'.format(args.dataset_name)
    # model_dir = os.path.join(data_dir, 'model_{}_{}_{}_{}'.format(args.dataset_name, args.neg_sampling_strategy,
    #                                                             args.classifier, args.validation_strategy))
    # model.load_state_dict(torch.load(model_dir))

    if args.neg_sampling_strategy == 'RANDOM':
        sample_dir = os.path.join(data_dir, 'train_random.txt')
    if args.neg_sampling_strategy == 'RANDOM_GO':
        sample_dir = os.path.join(data_dir, f'train_random_GO_{args.GO_sim}.txt')
    if args.neg_sampling_strategy == 'RANDOM_distance':
        sample_dir = os.path.join(data_dir, f'train_random_distance_{args.distance}.txt')
    if args.neg_sampling_strategy == 'RANDOM_subcellular':
        sample_dir = os.path.join(data_dir, f'train_random_subcellular.txt')
    if args.neg_sampling_strategy == 'DDB':
        sample_dir = os.path.join(data_dir, 'train_DDB.txt')
    if args.neg_sampling_strategy == 'DDB_GO':
        sample_dir = os.path.join(data_dir, f'train_DDB_GO_{args.GO_sim}.txt')
    if args.neg_sampling_strategy == 'DDB_subcellular':
        sample_dir = os.path.join(data_dir, 'train_DDB_subcellular.txt')
    if args.neg_sampling_strategy == 'DDB_distance':
        sample_dir = os.path.join(data_dir, f'train_DDB_distance_{args.distance}.txt')
    train_pair_loader = PairLoader(args, sample_dir)
    train_dataset = [data for data in train_pair_loader]

        # collate_fn=collate_fn)
    model.fit(train_dataset)

def val_c1c2c3(args, val_or_test):
    data_dir = 'dataset/{}'.format(args.dataset_name)
    if val_or_test == 'train':
        sample_dir = os.path.join(data_dir, 'c1_random.txt')
    elif val_or_test == 'val':
        sample_dir = os.path.join(data_dir, 'c2.txt')
    else:
        sample_dir = os.path.join(data_dir, 'c3.txt')

    val_loader = PairLoader(args, sample_dir)
    val_loader = [data for data in val_loader]

    # collate_fn=collate_fn)
    model = AE(args).to(args.device)
    model_dir = os.path.join(data_dir, 'model_{}_{}_{}_{}'.format(args.dataset_name, args.neg_sampling_strategy,
                                                                args.classifier, args.validation_strategy))
    model.load_state_dict(torch.load(model_dir))

    sample_available, val_true, val_pre = model.predict(val_loader)
    # val_true = np.vstack(val_true).squeeze(-1)
    # val_pre = np.vstack(val_pre)[:, 1]
    print('Samples: {}, val_AUC: {}'.format(len(sample_available), roc_auc_score(val_true, val_pre)))
    data_save = pd.DataFrame({'pair':sample_available, 'label':val_true, 'score':val_pre})
    predict_score_dir = os.path.join(data_dir, 'predicted_{}_{}_{}_{}'.format(args.dataset_name, args.neg_sampling_strategy,
                                                                args.classifier, args.validation_strategy))
    data_save.to_csv(predict_score_dir, header=None, index=None)

def val_cv(args, val_or_test):
    data_dir = 'dataset/{}'.format(args.dataset_name)
    if val_or_test == 'val':
        sample_dir = os.path.join(data_dir, 'test_random.txt')
    else:
        sample_dir = os.path.join(data_dir, 'test_random.txt')

    val_loader = PairLoader(args, sample_dir)
    val_loader = [data for data in val_loader]

    model = AE(args).to(args.device)
    model_dir = os.path.join(data_dir, 'model_{}_{}_{}_{}'.format(args.dataset_name, args.neg_sampling_strategy,
                                                                args.classifier, args.validation_strategy))
    model.load_state_dict(torch.load(model_dir))

    sample_available, val_true, val_pre = model.predict(val_loader)

    print('val_AUC: {}'.format(roc_auc_score(val_true, val_pre)))
    data_save = pd.DataFrame({'pair':sample_available, 'label':val_true, 'score':val_pre})
    predict_score_dir = os.path.join(data_dir, 'predicted_{}_{}_{}_{}'.format(args.dataset_name, args.neg_sampling_strategy,
                                                                args.classifier, args.validation_strategy))
    data_save.to_csv(predict_score_dir, header=None, index=None)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.validation_strategy == 'CV':
        train_cv(args)
        val_cv(args, 'val')
    if args.validation_strategy == 'c1c2c3':
        # train_c1c2c3(args)

        # train_c1c2c3(args)
        val_c1c2c3(args, 'val')
        val_c1c2c3(args, 'test')

from Arguments import parser
import os
from source.exteact_seq_feas import RNA_Init_Fea, Protein_Init_Fea, mol1fea_mol2fea
from prepare_dataset import make_dataset_c1v2c3, make_dataset_CV
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
import pandas as pd
def SEQ_RF(args):
    clf = RandomForestClassifier(n_estimators=500, criterion='entropy', random_state=0)
    return clf
def NOISE_RF(args):
    clf = RandomForestClassifier(n_estimators=500, criterion='entropy', random_state=0)
    return clf
def s_kernel(X, Y):
    mol_dimension = int(X.shape[1]/2)
    ae = euclidean_distances(X[:,:mol_dimension], Y[:,:mol_dimension], squared=True)
    bf = euclidean_distances(X[:,mol_dimension:], Y[:,mol_dimension:], squared=True)
    min1 = ae+bf
    af = euclidean_distances(X[:,:mol_dimension], Y[:,mol_dimension:], squared=True)
    be = euclidean_distances(X[:,mol_dimension:], Y[:,:mol_dimension], squared=True)
    min2 = af+be
    min = np.minimum(min1, min2)
    gama = 1.0 / (X.shape[1]*2)
    # gama=0.25
    return np.exp(-gama *min)

def SEQ_SVM(args):
    if args.task == 'PPI':
        clf = svm.SVC(probability=True)
    else:
        clf = svm.SVC(kernel=rbf_kernel, probability=True)
    # n_estimators = 10
    # clf = BaggingClassifier(clf, max_samples=1.0 / n_estimators, n_estimators=n_estimators,
    #                   n_jobs=-1)
    return clf

def train_c1c2c3(args):
    make_dataset_c1v2c3(args)
    # make_dataset_CV(args)
    data_dir = 'dataset/{}'.format(args.dataset_name)

    if args.neg_sampling_strategy == 'RANDOM':
        sample_dir = os.path.join(data_dir, 'c1_random.txt')
    if args.neg_sampling_strategy == 'RANDOM_GO':
        sample_dir = os.path.join(data_dir, f'c1_random_GO_{args.GO_sim}.txt')
    if args.neg_sampling_strategy == 'RANDOM_distance':
        sample_dir = os.path.join(data_dir, f'c1_random_distance_{args.distance}.txt')
    if args.neg_sampling_strategy == 'RANDOM_subcellular':
        sample_dir = os.path.join(data_dir, 'c1_random_subcellular.txt')
    if args.neg_sampling_strategy == 'DDB':
        sample_dir = os.path.join(data_dir, 'c1_DDB.txt')
    if args.neg_sampling_strategy == 'DDB_GO':
        sample_dir = os.path.join(data_dir, f'c1_DDB_GO_{args.GO_sim}.txt')
    if args.neg_sampling_strategy == 'DDB_subcellular':
        sample_dir = os.path.join(data_dir, 'c1_DDB_subcellular.txt')
    if args.neg_sampling_strategy == 'DDB_distance':
        sample_dir = os.path.join(data_dir, f'c1_DDB_distance_{args.distance}.txt')
    encoder = mol1fea_mol2fea(args)
    fea_1_dict = encoder.get_features(mol=1)
    if args.task == 'PPI':
        fea_2_dict = fea_1_dict
    else:
        fea_2_dict = encoder.get_features(mol=2)
    features = []
    labels = []
    with open(sample_dir, 'r') as pid:
        samples = pid.readlines()

    print('Loading pair-wise samples ({}) feature...'.format(len(samples)))
    for item in tqdm(samples):
        mol1, mol2, label = item.strip().split()
        if mol1 not in fea_1_dict.keys() or mol2 not in fea_2_dict.keys():
            continue
        features.append(np.hstack([fea_1_dict[mol1], fea_2_dict[mol2]]))
        labels.append(int(label))
    features = np.array(features)
    print('number of samples: ', len(labels))
    model = eval(args.classifier)(args)
    model.fit(features, labels)

    model_dir = os.path.join(data_dir, 'model_{}_{}_{}_{}'.format(args.dataset_name, args.neg_sampling_strategy, args.classifier, args.validation_strategy))
    with open(model_dir, 'wb') as pid:
        pickle.dump(model, pid)
    pres = model.predict_proba(features)
    print('{} AUC value:'.format('Training'), roc_auc_score(labels, pres[:,1]))

def train_cv(args):
    # make_dataset_(args)
    make_dataset_CV(args)
    data_dir = 'dataset/{}'.format(args.dataset_name)

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

    encoder = mol1fea_mol2fea(args)
    fea_1_dict = encoder.get_features(mol=1)
    if args.task == 'PPI':
        fea_2_dict = fea_1_dict
    else:
        fea_2_dict = encoder.get_features(mol=2)
    features = []
    labels = []
    with open(sample_dir, 'r') as pid:
        samples = pid.readlines()

    print('Loading pair-wise samples ({}) feature...'.format(len(samples)))
    for item in tqdm(samples):
        mol1, mol2, label = item.strip().split()
        if mol1 not in fea_1_dict.keys() or mol2 not in fea_2_dict.keys():
            continue
        features.append(np.hstack([fea_1_dict[mol1], fea_2_dict[mol2]]))
        labels.append(int(label))
    features = np.array(features)
    print('number of samples: ', len(labels))
    model = eval(args.classifier)(args)
    model.fit(features, labels)

    model_dir = os.path.join(data_dir, 'model_{}_{}_{}_{}'.format(args.dataset_name, args.neg_sampling_strategy, args.classifier, args.validation_strategy))
    with open(model_dir, 'wb') as pid:
        pickle.dump(model, pid)
    pres = model.predict_proba(features)
    print('{} AUC value:'.format('Training'), roc_auc_score(labels, pres[:,1]))

def val_c1c2c3(args, val_or_test):
    data_dir = 'dataset/{}'.format(args.dataset_name)
    if val_or_test == 'val':
        sample_dir = os.path.join(data_dir, 'c2.txt')
    else:
        sample_dir = os.path.join(data_dir, 'c3.txt')
    encoder = mol1fea_mol2fea(args)
    fea_1_dict = encoder.get_features(mol=1)
    if args.task == 'PPI':
        fea_2_dict = fea_1_dict
    else:
        fea_2_dict = encoder.get_features(mol=2)
    features = []
    features2 = []
    labels = []
    with open(sample_dir, 'r') as pid:
        samples = pid.readlines()

    print('Loading pair-wise samples ({}) feature...'.format(len(samples)))
    for item in tqdm(samples):
        mol1, mol2, label = item.strip().split()
        if mol1 not in fea_1_dict.keys() or mol2 not in fea_2_dict.keys():
            continue
        features.append(np.hstack([fea_1_dict[mol1], fea_2_dict[mol2]]))
        features2.append(np.hstack([fea_2_dict[mol2], fea_1_dict[mol1]]))
        labels.append(int(label))

    print('number of samples: ', len(labels))

    model_dir = os.path.join(data_dir, 'model_{}_{}_{}_{}'.format(args.dataset_name, args.neg_sampling_strategy, args.classifier, args.validation_strategy))
    with open(model_dir, 'rb') as pid:
        model = pickle.load(pid)
    pres = model.predict_proba(np.array(features))
    results_dir = os.path.join(data_dir, 'results.txt')
    with open(results_dir, 'a+') as pid:
        pid.write(val_or_test)
        pid.write(' ')
        pid.write(str(roc_auc_score(labels, pres[:,1])))
        pid.write('\n')
    print(val_or_test, ' ', roc_auc_score(labels, pres[:,1]))

def val_cv(args, val_or_test):
    data_dir = 'dataset/{}'.format(args.dataset_name)
    if val_or_test == 'val':
        sample_dir = os.path.join(data_dir, 'test_random.txt')
    else:
        sample_dir = os.path.join('dataset', 'test_n.txt')
    encoder = mol1fea_mol2fea(args)
    fea_1_dict = encoder.get_features(mol=1)
    if args.task == 'PPI':
        fea_2_dict = fea_1_dict
    else:
        fea_2_dict = encoder.get_features(mol=2)
    features = []
    features2 = []
    labels = []
    with open(sample_dir, 'r') as pid:
        samples = pid.readlines()

    print('Loading pair-wise samples ({}) feature...'.format(len(samples)))
    sample_available = []
    for item in tqdm(samples):
        mol1, mol2, label = item.strip().split()
        if mol1 not in fea_1_dict.keys() or mol2 not in fea_2_dict.keys():
            continue
        features.append(np.hstack([fea_1_dict[mol1], fea_2_dict[mol2]]))
        features2.append(np.hstack([fea_2_dict[mol2], fea_1_dict[mol1]]))
        labels.append(int(label))
        sample_available.append(mol1 +'\t'+mol2)
    print('number of samples: ', len(labels))

    model_dir = os.path.join(data_dir, 'model_{}_{}_{}_{}'.format(args.dataset_name, args.neg_sampling_strategy, args.classifier, args.validation_strategy))
    with open(model_dir, 'rb') as pid:
        model = pickle.load(pid)
    pres = model.predict_proba(np.array(features))
    data_save = pd.DataFrame({'pair':sample_available, 'label':labels, 'score':pres[:,1]})
    predict_score_dir = os.path.join(data_dir, 'predicted_{}_{}_{}_{}'.format(args.dataset_name, args.neg_sampling_strategy,
                                                                args.classifier, args.validation_strategy))
    data_save.to_csv(predict_score_dir, header=None, index=None)
    print(val_or_test, ' ', roc_auc_score(labels, pres[:,1]))

    # pres = pd.DataFrame(pres[:,1])
    # pres.to_csv(os.path.join(data_dir, 'predict_score.txt'), header=None, index=None)

if __name__ == '__main__':
    args = parser.parse_args()
    if args.validation_strategy == 'CV':
        train_cv(args)
        val_cv(args, 'val')
    if args.validation_strategy == 'c1c2c3':
        train_c1c2c3(args)
        val_c1c2c3(args, 'val')
        val_c1c2c3(args, 'test')
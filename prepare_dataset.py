import numpy as np
import random
from Arguments import parser
import networkx as nx
import os
import pickle
from tqdm import tqdm
from prepare_subcellular import DDB_sampling_subcellular, random_sampling_subcellular
from prepare_GO import DDB_sampling_GO, random_sampling_GO
from prepare_distance import DDB_sampling_distance, random_sampling_distance

def split_homo_c1c2c3(network, pairs):
    num_nodes = network.number_of_nodes()
    nodes = list(network.nodes)
    degree =[network.degree[i] for i in nodes]
    degree = np.array(degree)
    nodes_proba = degree / np.sum(degree)
    # test_nodes = np.random.choice(nodes, size=int(num_nodes*0.2), replace=False, )
    test_nodes = np.random.choice(nodes, size=int(num_nodes*0.1), replace=False, p=nodes_proba)

    c1_set = []
    c2_set = []
    c3_set = []
    print('spliting c1, c2, c3...')
    for pair in tqdm(pairs):
        node1, node2 = pair[0], pair[1]
        if node1 in test_nodes and node2 in test_nodes:
            c3_set.append(pair)
        elif node1 not in test_nodes and node2 not in test_nodes:
            c1_set.append(pair)
        else:
            c2_set.append(pair)
    print('c1_set: ', len(c1_set),'c2_set: ', len(c2_set),'c3_set: ', len(c3_set))
    return c1_set, c2_set, c3_set

def split_c1c2c3(network, pairs):

    partition = nx.algorithms.community.kernighan_lin_bisection(network)
    set1, set2 = partition
    # Create two subgraphs based on the partition
    subgraph2 = network.subgraph(set2)
    partition2 = nx.algorithms.community.kernighan_lin_bisection(subgraph2)
    set3, set4 = partition2

    c1_set = []
    c2_set = []
    c3_set = []
    print('spliting c1, c2, c3...')
    for pair in tqdm(pairs):
        node1, node2 = pair[0], pair[1]
        if node1 in set4 and node2 in set4:
            c3_set.append(pair)
        elif node1 not in set4 and node2 not in set4:
            c1_set.append(pair)
        else:
            c2_set.append(pair)
    return c1_set, c2_set, c3_set

def split_random(network, fraction_train = 0.9):
    edges = list(network.edges)
    random.shuffle(edges)
    train_edges = edges[0: int(len(edges)*fraction_train)]
    val_edges = edges[int(len(edges)*fraction_train):]
    return train_edges, val_edges

def random_sampling(args, network, fold=1):
    #pairs example
    #[[p1, p2], [p3, p4]...]
    pairs = network.edges
    print('Negative randomly sampling...')
    positive_pair = set()
    mol1_set = set()
    mol2_set = set()

    for item in pairs:
        positive_pair.add(item[0] +'\t'+ item[1])
        mol1_set.add(item[0])
        mol2_set.add(item[1])

    if args.task == 'PPI':
        mol1_set.update(mol2_set)

    random_pairs = set()
    mol1_set = list(mol1_set)
    mol2_set = list(mol2_set)
    sampling_num = len(pairs)*fold
    while len(random_pairs)!=sampling_num:
        if args.task == 'PPI':
            p1 = random.choice(mol1_set)
            p2 = random.choice(mol1_set)
        else:
            p1 = random.choice(mol1_set)
            p2 = random.choice(mol2_set)
        if p1 == p2:
            continue
        if p1+'\t'+p2 in positive_pair or p2+'\t'+p1 in positive_pair:
            continue
        if p1+'\t'+p2 in random_pairs or p2+'\t'+p1 in random_pairs:
            continue
        random_pairs.add(p1+'\t'+p2)
    neg_randoms = []
    for item in random_pairs:
        p1, p2 = item.split('\t')
        neg_randoms.append([p1, p2])
    return neg_randoms


def DDB_sampling(args, network):

    print('Negative DDB sampling...')
    pairs = list(network.edges)
    pairs = np.array(pairs)
    data_dir = 'dataset/{}'.format(args.dataset_name)
    if args.validation_strategy == 'CV':
        edge_degree_pkl = os.path.join(data_dir, 'edge_degree_.pkl')
    else:
        edge_degree_pkl = os.path.join(data_dir, 'edge_degree_train.pkl')
    with open(edge_degree_pkl, 'rb') as pid:
        edge_degree = pickle.load(pid)

    edge_degree=dict(sorted(edge_degree.items(), key=lambda d: d[0]))
    for item in edge_degree.keys():
        a = list(edge_degree[item])
        random.shuffle(a)
        edge_degree[item] = a
    degree_keys = list(edge_degree.keys())
    data = []
    for item in pairs:
        data.append(item[0] + '\t'+item[1])
        data.append(item[1] + '\t'+item[0])
    data = set(data)
    neg_edges=set()
    for item in tqdm(pairs):
        # item=item.strip().split(' ')
        key=[(network.degree[item[0]] + network.degree[item[1]], 'middle'), ]
        while True:
            if key[0][0] not in edge_degree:
                if len(key) > 1:
                    del (key[0])
                    continue
                else:
                    break
            value=edge_degree[key[0][0]]
            # random.shuffle(value)
            flag=False
            for i in range(len(value)):
                if value[i] not in data:
                    neg_edges.add(value[i])
                    data.add(value[i])
                    data.add(value[i].split('\t')[1] + '\t' + value[i].split('\t')[0])
                    flag=True
                    break
                else:
                    continue
            if flag:
                break
            position=degree_keys.index(key[0][0])
            if key[0][1] == 'left' and position != 0:
                key.append((degree_keys[position - 1], 'left'))

            if key[0][1] == 'right' and position != len(degree_keys) - 1:
                key.append((degree_keys[position + 1], 'right'))

            if key[0][1] == 'middle':
                if position != 0:
                    key.append((degree_keys[position - 1], 'left'))
                if position != len(degree_keys) - 1:
                    key.append((degree_keys[position + 1], 'right'))
            del (key[0])

    new_neg_edges = []
    for item in neg_edges:
        m1, m2 = item.split('\t')
        new_neg_edges.append([m1, m2])
    return new_neg_edges


def cal_edge_degree(args, network, name=''):
    data_dir = 'dataset/{}'.format(args.dataset_name)
    if os.path.exists(os.path.join(data_dir, 'edge_degree_{}.pkl'.format(name))):
        return
    print('Edge degree generating...')
    pairs = list(network.edges)
    pairs = np.array(pairs)
    if args.task == 'PPI':
        mo1_set = list(network.nodes)
        mo2_set = list(network.nodes)
    else:
        pairs = np.array(pairs)
        mo1_set = list(set(pairs[:,0].tolist()))
        mo2_set = list(set(pairs[:,1].tolist()))

    edge_degree = {}
    number = len(mo1_set)
    for mo1_index in tqdm(range(number)):
        mo1 = mo1_set[mo1_index]
        for mo2 in mo2_set:
            key = network.degree[mo1] + network.degree[mo2]
            if key not in edge_degree.keys():
                edge_degree[key] = set()
            edge_degree[key].add(mo1 + '\t' + mo2)
    with open(os.path.join(data_dir, 'edge_degree_{}.pkl'.format(name)), 'wb') as pid:
        pickle.dump(edge_degree, pid)


def make_file_string(pairs, label='1'):
    out = []
    for item in pairs:
        out.append(item[0] + ' ' + item[1] + ' ' + label + '\n')
    return out

def make_dataset_c1v2c3(args):
    data_dir = 'dataset/{}'.format(args.dataset_name)
    if os.path.exists(os.path.join(data_dir, 'c1_random.txt')):
        return
    with open(os.path.join(data_dir, 'network.txt'), 'r') as pid:
        lines = pid.readlines()
    pos = []
    for line in lines:
        p1,p2 = line.strip().split()
        pos.append((p1, p2))
    network = nx.Graph()
    network.add_edges_from(pos)
    cal_edge_degree(args, network)

    c1_set, c2_set, c3_set = split_c1c2c3(network, pos)#c1_set:train+val. c2_set:test. and c3_set:test

    print(len(c1_set), len(c2_set), len(c3_set))
    c1_network = nx.DiGraph()
    c1_network.add_edges_from(c1_set)
    cal_edge_degree(args, c1_network, 'train')

    c2_network = nx.DiGraph()
    c2_network.add_edges_from(c2_set)
    c3_network = nx.DiGraph()
    c3_network.add_edges_from(c3_set)
    c1_neg_random = random_sampling(args,c1_network, 1)
    c1_neg_random_distance = random_sampling_distance(args,c1_network, 1)

    if args.if_GO_Sub:
        c1_neg_random_GO = random_sampling_GO(args, c1_network, 1)
        c1_neg_random_subcellular = random_sampling_subcellular(args, c1_network, 1)
        c1_neg_DDB_GO = DDB_sampling_GO(args, c1_network)
        c1_neg_DDB_subcellular = DDB_sampling_subcellular(args, c1_network)

    c1_neg_DDB = DDB_sampling(args, c1_network)
    c1_neg_DDB_distance = DDB_sampling_distance(args, c1_network)

    c2_neg = random_sampling(args, c2_network, 1)
    c3_neg = random_sampling(args, c3_network, 1)
    # c3_neg2 = DDB_sampling(args, c3_network, c3_set)
    # c3_neg = c3_neg2+c3_neg
    random.shuffle(c1_set)
    random.shuffle(c1_neg_random)
    random.shuffle(c1_neg_random_distance)
    random.shuffle(c1_neg_DDB)
    if args.if_GO_Sub:
        random.shuffle(c1_neg_random_GO)
        random.shuffle(c1_neg_random_subcellular)
        random.shuffle(c1_neg_DDB_GO)
        random.shuffle(c1_neg_DDB_subcellular)

    random.shuffle(c1_neg_DDB_distance)
    c1_random = make_file_string(c1_set, '1') + make_file_string(c1_neg_random, '0')
    c1_random_distance = make_file_string(c1_set, '1') + make_file_string(c1_neg_random_distance, '0')
    c1_DDB = make_file_string(c1_set, '1') + \
             make_file_string(c1_neg_DDB[:int(len(c1_neg_DDB)*0.8)], '0') + \
             make_file_string(c1_neg_DDB[:int(len(c1_neg_random)*0.2)], '0')

    if args.if_GO_Sub:
        c1_random_GO = make_file_string(c1_set, '1') + make_file_string(c1_neg_random_GO, '0')
        c1_random_subcellular = make_file_string(c1_set, '1') + make_file_string(c1_neg_random_subcellular, '0')
        c1_DDB_GO = make_file_string(c1_set, '1') + \
                    make_file_string(c1_neg_DDB_GO[:int(len(c1_neg_DDB_GO) * 0.8)], '0') + \
                    make_file_string(c1_neg_DDB_GO[:int(len(c1_neg_random) * 0.2)], '0')
        c1_DDB_subcellular = make_file_string(c1_set, '1') + \
                             make_file_string(c1_neg_DDB_subcellular[:int(len(c1_neg_DDB_subcellular) * 0.8)], '0') + \
                             make_file_string(c1_neg_DDB_subcellular[:int(len(c1_neg_random) * 0.2)], '0')

    c1_DDB_distance = make_file_string(c1_set, '1') + \
                make_file_string(c1_neg_DDB_distance[:int(len(c1_neg_DDB_distance) * 0.8)], '0') + \
                make_file_string(c1_neg_DDB_distance[:int(len(c1_neg_random) * 0.2)], '0')

    c2 = make_file_string(c2_set, '1') + make_file_string(c2_neg, '0')
    c3 = make_file_string(c3_set, '1') + make_file_string(c3_neg, '0')

    with open(os.path.join(data_dir, 'c1_random.txt'), 'w') as pid:
        pid.writelines(c1_random)
    with open(os.path.join(data_dir, f'c1_random_distance_{args.distance}.txt'), 'w') as pid:
        pid.writelines(c1_random_distance)
    with open(os.path.join(data_dir, 'c1_DDB.txt'), 'w') as pid:
        pid.writelines(c1_DDB)
    if args.if_GO_Sub:
        with open(os.path.join(data_dir, f'c1_random_GO_{args.GO_sim}.txt'), 'w') as pid:
            pid.writelines(c1_random_GO)
        with open(os.path.join(data_dir, 'c1_random_subcellular.txt'), 'w') as pid:
            pid.writelines(c1_random_subcellular)
        with open(os.path.join(data_dir, f'c1_DDB_GO_{args.GO_sim}.txt'), 'w') as pid:
            pid.writelines(c1_DDB_GO)
        with open(os.path.join(data_dir, 'c1_DDB_subcellular.txt'), 'w') as pid:
            pid.writelines(c1_DDB_subcellular)
    with open(os.path.join(data_dir, f'c1_DDB_distance_{args.distance}.txt'), 'w') as pid:
        pid.writelines(c1_DDB_distance)
    with open(os.path.join(data_dir, 'c2.txt'), 'w') as pid:
        pid.writelines(c2)
    with open(os.path.join(data_dir, 'c3.txt'), 'w') as pid:
        pid.writelines(c3)

def make_dataset_CV(args):
    data_dir = 'dataset/{}'.format(args.dataset_name)

    if os.path.exists(os.path.join(data_dir, 'train_random.txt')):
        return
    with open(os.path.join(data_dir, 'network.txt'), 'r') as pid:
        lines = pid.readlines()
    pos = []
    for line in lines:
        p1,p2 = line.strip().split()
        pos.append((p1, p2))
    network = nx.DiGraph()
    network.add_edges_from(pos)
    cal_edge_degree(args, network)
    # exit()
    train_set, test_set = split_random(network, fraction_train=args.CV_frac)# return list, list

    train_network = nx.DiGraph()
    train_network.add_edges_from(train_set)


    cal_edge_degree(args, train_network, 'train')

    test_network = nx.DiGraph()
    test_network.add_edges_from(test_set)

    neg_random = random_sampling(args, network, 1) #return list
    neg_random_distance = random_sampling_distance(args, network, 1)
    if args.if_GO_Sub:
        neg_random_GO = random_sampling_GO(args, network, 1)
        neg_random_subcellular = random_sampling_subcellular(args, network, 1)
        train_neg_DDB_GO = DDB_sampling_GO(args, network)
        train_neg_DDB_subcellular = DDB_sampling_subcellular(args, network)
        train_neg_random_GO = neg_random_GO[:int(len(neg_random_GO) * args.CV_frac)]
        train_neg_random_subcellular = neg_random_subcellular[:int(len(neg_random_subcellular) * args.CV_frac)]
    train_neg_random = neg_random[:int(len(neg_random)*args.CV_frac)]
    train_neg_random_distance = neg_random_distance[:int(len(neg_random_distance)*args.CV_frac)]
    train_neg_DDB = DDB_sampling(args, network)
    ##
    train_neg_DDB_distance = DDB_sampling_distance(args, network)
    # test_neg_random = random_sampling(args, test_network, 1)
    test_neg_random = neg_random[int(len(neg_random)*args.CV_frac):]
    # test_neg_DDB = DDB_sampling(args, test_network)


    train_random = make_file_string(train_set, '1') + make_file_string(train_neg_random, '0')
    train_random_distance = make_file_string(train_set, '1') + make_file_string(train_neg_random_distance, '0')
    if args.if_GO_Sub:
        train_random_subcellular = make_file_string(train_set, '1') + make_file_string(train_neg_random_subcellular, '0')
        train_random_GO = make_file_string(train_set, '1') + make_file_string(train_neg_random_GO, '0')
        train_DDB_GO = make_file_string(train_set, '1') + make_file_string(train_neg_DDB_GO, '0')
        train_DDB_subcellular = make_file_string(train_set, '1') + make_file_string(train_neg_DDB_subcellular, '0')
    test_random = make_file_string(test_set, '1') + make_file_string(test_neg_random, '0')

    train_DDB = make_file_string(train_set, '1') + make_file_string(train_neg_DDB, '0')
    ##
    train_DDB_distance = make_file_string(train_set, '1') + make_file_string(train_neg_DDB_distance, '0')
    # test_DDB = make_file_string(test_set, '1') + make_file_string(test_neg_random, '0')



    with open(os.path.join(data_dir, 'train_random.txt'), 'w') as pid:
        pid.writelines(train_random)
    with open(os.path.join(data_dir, f'train_random_distance_{args.distance}.txt'), 'w') as pid:
        pid.writelines(train_random_distance)
    if args.if_GO_Sub:
        with open(os.path.join(data_dir, f'train_random_GO_{args.GO_sim}.txt'), 'w') as pid:
            pid.writelines(train_random_GO)
        with open(os.path.join(data_dir, 'train_random_subcellular.txt'), 'w') as pid:
            pid.writelines(train_random_subcellular)
        with open(os.path.join(data_dir, f'train_DDB_GO_{args.GO_sim}.txt'), 'w') as pid:
            pid.writelines(train_DDB_GO)
        with open(os.path.join(data_dir, 'train_DDB_subcellular.txt'), 'w') as pid:
            pid.writelines(train_DDB_subcellular)
    with open(os.path.join(data_dir, 'test_random.txt'), 'w') as pid:
        pid.writelines(test_random)
    with open(os.path.join(data_dir, 'train_DDB.txt'), 'w') as pid:
        pid.writelines(train_DDB)
    ####
    with open(os.path.join(data_dir, f'train_DDB_distance_{args.distance}.txt'), 'w') as pid:
        pid.writelines(train_DDB_distance)


if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = 'dataset/{}'.format(args.dataset_name)

    if os.path.exists(os.path.join(data_dir, 'train_random.txt')):
        print("yet")


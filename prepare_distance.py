import random
from Arguments import parser
import networkx as nx
import os
import pickle
from tqdm import tqdm

def random_sampling_distance(args, network, fold=1):
    distance_threshold = args.distance
    # pairs example
    # [[p1, p2], [p3, p4]...]
    pairs = network.edges
    print('Negative randomly (with distance) sampling...')

    positive_pair = set()
    mol1_set = set()
    mol2_set = set()

    for item in pairs:
        positive_pair.add(item[0] + '\t' + item[1])
        mol1_set.add(item[0])
        mol2_set.add(item[1])

    if args.task == 'PPI':
        mol1_set.update(mol2_set)

    random_pairs = set()
    mol1_set = list(mol1_set)
    mol2_set = list(mol2_set)
    sampling_num = len(pairs) * fold

    data_dir = 'dataset/{}'.format(args.dataset_name)
    distance_file = os.path.join(data_dir, 'edge_distances.pkl')

    if not os.path.exists(distance_file):
        print('Computing node distances...')
        distances = compute_node_distances(network)
        with open(distance_file, 'wb') as f:
            pickle.dump(distances, f)
    else:
        with open(distance_file, 'rb') as f:
            distances = pickle.load(f)

    while len(random_pairs) != sampling_num:
        if args.task == 'PPI':
            p1 = random.choice(mol1_set)
            p2 = random.choice(mol1_set)
        else:
            p1 = random.choice(mol1_set)
            p2 = random.choice(mol2_set)

        if p1 == p2:
            continue
        if p1 + '\t' + p2 in positive_pair or p2 + '\t' + p1 in positive_pair:
            continue
        if p1 + '\t' + p2 in random_pairs or p2 + '\t' + p1 in random_pairs:
            continue


        # if distances.get((p1, p2), float('inf')) >= distance_threshold and distances.get((p1, p2)) != float('inf'):
        if distances.get((p1, p2), float('inf')) >= distance_threshold:
            random_pairs.add(p1 + '\t' + p2)

    neg_randoms = []
    for item in random_pairs:
        p1, p2 = item.split('\t')
        neg_randoms.append([p1, p2])

    return neg_randoms


def DDB_sampling_distance(args, network):
    distance_threshold = args.distance
    print('Negative DDB sampling with distance threshold...')

    pairs = list(network.edges)
    data_dir = 'dataset/{}'.format(args.dataset_name)


    if args.validation_strategy == 'CV':
        edge_degree_pkl = os.path.join(data_dir, 'edge_degree_.pkl')
    else:
        edge_degree_pkl = os.path.join(data_dir, 'edge_degree_train.pkl')


    with open(edge_degree_pkl, 'rb') as pid:
        edge_degree = pickle.load(pid)


    distance_file = os.path.join(data_dir, 'edge_distances.pkl')
    if not os.path.exists(distance_file):
        print('Computing node distances...')
        distances = compute_node_distances(network)
        with open(distance_file, 'wb') as f:
            pickle.dump(distances, f)
    else:
        with open(distance_file, 'rb') as f:
            distances = pickle.load(f)


    edge_degree = dict(sorted(edge_degree.items(), key=lambda d: d[0]))


    for item in edge_degree.keys():
        edge_degree[item] = list(edge_degree[item])
        random.shuffle(edge_degree[item])

    degree_keys = list(edge_degree.keys())
    data = set()
    for item in pairs:
        data.add(item[0] + '\t' + item[1])
        data.add(item[1] + '\t' + item[0])

    neg_edges = set()


    num_selected = 0
    num_skipped_due_to_distance = 0
    num_degree_mismatch = 0
    same_degree_accepted = 0
    accepted_edges = 0
    total_attempts = 0


    with tqdm(total=len(pairs)) as pbar:
        for item in pairs:
            pos_degree_sum = network.degree[item[0]] + network.degree[item[1]]
            key = [(pos_degree_sum, 'middle')]
            same_degree = True

            while True:
                if key[0][0] not in edge_degree:
                    if len(key) > 1:
                        del key[0]
                        continue
                    else:
                        break

                total_attempts += 1
                value = edge_degree[key[0][0]]

                flag = False
                for i in range(len(value)):
                    node_pair = value[i]
                    node1, node2 = node_pair.split('\t')


                    if node1 == node2:
                        continue


                    if node_pair not in data and distances.get((node1, node2), float('inf')) >= distance_threshold:
                    # if node_pair not in data and distances.get((node1, node2), float('inf')) >= distance_threshold and distances.get((node1, node2)) != float('inf'):
                        neg_edges.add(node_pair)
                        data.add(node_pair)
                        data.add(f"{node2}\t{node1}")
                        accepted_edges += 1


                        if key[0][0] == pos_degree_sum:
                            same_degree_accepted += 1
                        else:
                            num_degree_mismatch += 1

                        flag = True
                        break
                    else:
                        num_skipped_due_to_distance += 1

                if flag:
                    break


                position = degree_keys.index(key[0][0])
                if key[0][1] == 'left' and position != 0:
                    key.append((degree_keys[position - 1], 'left'))
                if key[0][1] == 'right' and position != len(degree_keys) - 1:
                    key.append((degree_keys[position + 1], 'right'))
                if key[0][1] == 'middle':
                    if position != 0:
                        key.append((degree_keys[position - 1], 'left'))
                    if position != len(degree_keys) - 1:
                        key.append((degree_keys[position + 1], 'right'))
                del key[0]

            pbar.update(1)


    new_neg_edges = []
    for item in neg_edges:
        m1, m2 = item.split('\t')
        new_neg_edges.append([m1, m2])

    # 输出统计信息
    print(f"distance_threshold：{distance_threshold}")
    print(f"accepted_edges： {accepted_edges}")
    print(f"num_skipped_due_to_distance： {num_skipped_due_to_distance} ")

    if accepted_edges > 0:
        same_degree_percentage = (same_degree_accepted / accepted_edges) * 100
        mismatch_degree_percentage = (num_degree_mismatch / accepted_edges) * 100
        print(f"same_degree_percentage： {same_degree_percentage:.2f}%")
        print(f"mismatch_degree_percentage： {mismatch_degree_percentage:.2f}%")

    return new_neg_edges


def compute_node_distances(network):
    distances = {}
    for node1 in network.nodes:
        for node2 in network.nodes:
            if node1 != node2:
                try:
                    distances[(node1, node2)] = nx.shortest_path_length(network, node1, node2)
                except nx.NetworkXNoPath:
                    distances[(node1, node2)] = float('inf')
    return distances

if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = 'dataset/{}'.format(args.dataset_name)
    with open(os.path.join(data_dir, 'network.txt'), 'r') as pid:
        lines = pid.readlines()
    pos = []
    for line in lines:
        p1, p2 = line.strip().split()
        pos.append((p1, p2))
    network = nx.Graph()
    network.add_edges_from(pos)
    distance_file = os.path.join(data_dir, 'edge_distances.pkl')
    if not os.path.exists(distance_file):
        print('Computing node distances...')
        distances = compute_node_distances(network)
        with open(distance_file, 'wb') as f:
            pickle.dump(distances, f)
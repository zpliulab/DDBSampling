import random
import requests
from Arguments import parser
import networkx as nx
from go_utils import get_go_dag
import os
import pickle
from tqdm import tqdm
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def get_protein_subcellular_location(uniprot_ids):
    subcellular_locations = {}

    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        method_whitelist=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)

    for uniprot_id in tqdm(uniprot_ids, desc="Fetching subcellular locations"):
        uniprot_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
        try:
            response = http.get(uniprot_url)
            if response.status_code == 200:
                data = response.json()
                locations = []
                if "comments" in data:
                    for comment in data["comments"]:
                        if comment["commentType"] == "SUBCELLULAR LOCATION":
                            for subcell in comment.get("subcellularLocations", []):
                                locations.append(subcell["location"]["value"])

                subcellular_locations[uniprot_id] = locations
            else:
                print(f"Failed to fetch data for Uniprot ID: {uniprot_id}")

            time.sleep(0.01)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {uniprot_id}: {e}")

    return subcellular_locations

def load_or_fetch_subcellular_locations(uniprot_ids, cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    subcellular_locations = get_protein_subcellular_location(uniprot_ids)

    with open(cache_file, 'wb') as f:
        pickle.dump(subcellular_locations, f)

    return subcellular_locations


def normalize_subcellular_locations(locations):
    """
    Split the complex subcellular localization into individual units, such as 'Cytoplasm, cytoskeleton' into 'Cytoplasm' and 'cytoskeleton'.
    """
    normalized_locations = set()
    for loc in locations:
        sub_locs = [s.strip() for s in loc.split(',')]
        normalized_locations.update(sub_locs)
    return normalized_locations


def compute_subcellular_location(uniprot_locations, network):
    subcellular_edges = {}

    nodes = list(network.nodes)
    total_pairs = len(nodes) * (len(nodes) - 1) // 2
    progress_bar = tqdm(total=total_pairs, desc="Computing subcellular locations")

    for i, node1 in enumerate(nodes):
        for node2 in nodes[i+1:]:
            if node1 != node2:
                locations1 = normalize_subcellular_locations(uniprot_locations.get(node1, []))
                locations2 = normalize_subcellular_locations(uniprot_locations.get(node2, []))

                # print(locations1)
                # print(locations2)

                if not locations1 or not locations2:
                    subcellular_edges[(node1, node2)] = 2
                elif locations1 & locations2:
                    subcellular_edges[(node1, node2)] = 1
                else:
                    subcellular_edges[(node1, node2)] = 0

                # print(f"{node1}-{node2}: {subcellular_edges[(node1, node2)]}")

                progress_bar.update(1)

    progress_bar.close()
    return subcellular_edges


def random_sampling_subcellular(args, network, fold=1):
    # pairs example
    # [[p1, p2], [p3, p4]...]
    pairs = network.edges
    print('Negative randomly (with subcellular) sampling...')

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
    subcellular_file = os.path.join(data_dir, 'edge_subcellular.pkl')
    if not os.path.exists(subcellular_file):
        print('Computing node subcellular locations...')
        uniprot_ids = [node for node in network.nodes]
        uniprot_locations = load_or_fetch_subcellular_locations(uniprot_ids,
                                                                os.path.join(data_dir, 'uniprot_subcellular.pkl'))
        subcellular_locations = compute_subcellular_location(uniprot_locations, network)
        with open(subcellular_file, 'wb') as f:
            pickle.dump(subcellular_locations, f)
    else:
        with open(subcellular_file, 'rb') as f:
            subcellular_locations = pickle.load(f)

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

        if subcellular_locations.get((p1, p2), 1) == 0:
            random_pairs.add(p1 + '\t' + p2)

    neg_randoms = []
    for item in random_pairs:
        p1, p2 = item.split('\t')
        neg_randoms.append([p1, p2])

    return neg_randoms


def DDB_sampling_subcellular(args, network):
    print('Negative DDB sampling with subcellular location constraint...')

    pairs = list(network.edges)
    data_dir = 'dataset/{}'.format(args.dataset_name)

    if args.validation_strategy == 'CV':
        edge_degree_pkl = os.path.join(data_dir, 'edge_degree_.pkl')
    else:
        edge_degree_pkl = os.path.join(data_dir, 'edge_degree_train.pkl')

    with open(edge_degree_pkl, 'rb') as pid:
        edge_degree = pickle.load(pid)

    subcellular_file = os.path.join(data_dir, 'edge_subcellular.pkl')
    if not os.path.exists(subcellular_file):
        print('Computing node subcellular locations...')
        uniprot_ids = [node for node in network.nodes]
        uniprot_locations = load_or_fetch_subcellular_locations(uniprot_ids, os.path.join(data_dir, 'uniprot_subcellular.pkl'))
        subcellular_locations = compute_subcellular_location(uniprot_locations, network)
        with open(subcellular_file, 'wb') as f:
            pickle.dump(subcellular_locations, f)
    else:
        with open(subcellular_file, 'rb') as f:
            subcellular_locations = pickle.load(f)

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
    num_skipped_due_to_subcellular = 0
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


                    if node_pair not in data and subcellular_locations.get((node1, node2), 1) == 0:
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
                        num_skipped_due_to_subcellular += 1

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
    print(f"accepted_edges： {accepted_edges}")
    print(f"num_skipped_due_to_subcellular： {num_skipped_due_to_subcellular} ")

    if accepted_edges > 0:
        same_degree_percentage = (same_degree_accepted / accepted_edges) * 100
        mismatch_degree_percentage = (num_degree_mismatch / accepted_edges) * 100
        print(f"same_degree_percentage： {same_degree_percentage:.2f}%")
        print(f"mismatch_degree_percentage： {mismatch_degree_percentage:.2f}%")

    return new_neg_edges

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
    subcellular_file = os.path.join(data_dir, 'edge_subcellular.pkl')
    if not os.path.exists(subcellular_file):
        print('Computing node subcellular locations...')
        uniprot_ids = [node for node in network.nodes]
        uniprot_locations = load_or_fetch_subcellular_locations(uniprot_ids,
                                                                os.path.join(data_dir, 'uniprot_subcellular.pkl'))
        subcellular_locations = compute_subcellular_location(uniprot_locations, network)
        with open(subcellular_file, 'wb') as f:
            pickle.dump(subcellular_locations, f)
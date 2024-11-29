import random
import requests
from Arguments import parser
import networkx as nx
from go_utils import get_go_dag
from bioservices import UniProt
from goatools.semantic import lin_sim, TermCounts, resnik_sim
import os
from goatools.associations import dnld_assc, read_gaf
import gzip
import shutil
import re
import itertools
import pickle
from tqdm import tqdm

# go_basic.obo
def download_go_obo(save_path="go_basic.obo"):
    url = "http://current.geneontology.org/ontology/go-basic.obo"
    response = requests.get(url)

    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded go-basic.obo to {save_path}")
    else:
        print("Failed to download go-basic.obo")
        response.raise_for_status()


go_obo_path = "go_basic.obo"
if not os.path.exists(go_obo_path):
    download_go_obo(go_obo_path)



def compute_and_store_go_similarities(uniprot_ids, output_file):
    # GO DAG
    print("Calculate the GO similarity of all edges：")
    go_dag = get_go_dag()
    associations = get_associations()
    termcounts = get_term_counts(go_dag, associations)
    # GO
    go_annotations = get_go_annotations(uniprot_ids)

    go_similarities = {}

    # tqdm
    protein_pairs = itertools.combinations(uniprot_ids, 2)
    num_pairs = len(list(protein_pairs))
    protein_pairs = itertools.combinations(uniprot_ids, 2)

    for protein1, protein2 in tqdm(protein_pairs, total=num_pairs, desc="Computing GO Similarities"):
        if (protein2, protein1) not in go_similarities:
            sim = compute_go_similarity(protein1, protein2, go_annotations, go_dag, termcounts)
            go_similarities[(protein1, protein2)] = sim
            go_similarities[(protein2, protein1)] = sim

    with open(output_file, 'wb') as f:
        pickle.dump(go_similarities, f)
    print(f"GO similarity data saved in {output_file}")


# GAF
def get_associations():
    # GAF
    gaf_url = "http://current.geneontology.org/annotations/goa_human.gaf.gz"
    gz_file = os.path.join(os.getcwd(), "goa_human.gaf.gz")
    gaf_file = os.path.join(os.getcwd(), "goa_human.gaf")

    # if not exist
    if not os.path.exists(gz_file):
        import urllib.request
        print(f"Downloading {gaf_url}...")
        urllib.request.urlretrieve(gaf_url, gz_file)
        print("Download completed.")

    if not os.path.exists(gaf_file):
        print("Decompressing the .gz file...")
        with gzip.open(gz_file, 'rb') as f_in:
            with open(gaf_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print("Decompression completed.")

    print("Reading GAF file...")
    associations = read_gaf(gaf_file)
    print("GAF file read completed.")
    return associations

def get_go_annotations(uniprot_ids):
    print("Get go annotations：")
    u = UniProt()
    go_annotations = {}

    for uniprot_id in tqdm(uniprot_ids, desc="Fetching GO Annotations"):
        result = u.search(uniprot_id, columns="id,go")
        lines = result.split("\n")

        # print(f"UniProt result for {uniprot_id}:\n{result}")

        if len(lines) > 1 and "\t" in lines[1]:
            go_fields = lines[1].split("\t")
            if len(go_fields) > 1:
                go_terms = go_fields[1].split("; ")
                go_annotations[uniprot_id] = set(go_terms)
            else:
                go_annotations[uniprot_id] = set()
        else:
            go_annotations[uniprot_id] = set()

    return go_annotations

# term counts
def get_term_counts(go_dag, associations):
    return TermCounts(go_dag, associations)


def compute_go_similarity(protein1, protein2, go_annotations, go_dag, termcounts):
    go_set1 = go_annotations.get(protein1, set())
    go_set2 = go_annotations.get(protein2, set())

    # print(go_set1)
    # print(go_set2)

    go_list1 = [re.search(r'\[GO:\d+\]', go).group(0)[1:-1] for go in go_set1 if re.search(r'\[GO:\d+\]', go)]
    go_list2 = [re.search(r'\[GO:\d+\]', go).group(0)[1:-1] for go in go_set2 if re.search(r'\[GO:\d+\]', go)]

    # print(go_list1)
    # print(go_list2)

    if not go_list1 or not go_list2:
        return 2.0  # If the GO annotation does not exist, assume high similarity and avoid choosing a negative sample

    # Lin's
    sim = compute_go_set_similarity(go_list1, go_list2, go_dag, termcounts)
    return sim


def compute_go_set_similarity(go_list1, go_list2, go_dag, termcounts, method='max'):
    similarities = []
    for go_id1 in go_list1:
        for go_id2 in go_list2:
            try:
                sim = lin_sim(go_id1, go_id2, go_dag, termcounts)
                if sim is not None:
                    similarities.append(sim)
            except KeyError:
                continue

    if not similarities:
        return None

    if method == 'average':
        return sum(similarities) / len(similarities)
    elif method == 'max':
        return max(similarities)
    elif method == 'min':
        return min(similarities)
    else:
        raise ValueError(f"Unknown method {method}")


def random_sampling_GO(args, network, fold=1):
    print('Negative random sampling with GO similarity filtering...')

    pairs = network.edges
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
    uniprot_ids = list(set([item[0] for item in pairs] + [item[1] for item in pairs]))
    go_similarities_pkl = os.path.join(data_dir, 'go_similarities_.pkl')

    if not os.path.exists(go_similarities_pkl):
        print("Computing GO similarities...")
        compute_and_store_go_similarities(uniprot_ids, go_similarities_pkl)

    with open(go_similarities_pkl, 'rb') as f:
        go_similarities = pickle.load(f)

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

        go_sim = go_similarities.get((p1, p2), 1.0)
        if go_sim is not None and go_sim <= args.GO_sim:
            random_pairs.add(p1 + '\t' + p2)

    neg_randoms = []
    for item in random_pairs:
        p1, p2 = item.split('\t')
        neg_randoms.append([p1, p2])

    print(f"Negative samples generated: {len(neg_randoms)}")
    return neg_randoms


def DDB_sampling_GO(args, network):
    go_similarity_threshold = args.GO_sim
    print('Negative DDB sampling with GO similarity filtering...')
    pairs = list(network.edges)
    data_dir = 'dataset/{}'.format(args.dataset_name)

    if args.validation_strategy == 'CV':
        edge_degree_pkl = os.path.join(data_dir, 'edge_degree_.pkl')
    else:
        edge_degree_pkl = os.path.join(data_dir, 'edge_degree_train.pkl')

    with open(edge_degree_pkl, 'rb') as pid:
        edge_degree = pickle.load(pid)

    edge_degree = dict(sorted(edge_degree.items(), key=lambda d: d[0]))
    for item in edge_degree.keys():
        a = list(edge_degree[item])
        random.shuffle(a)
        edge_degree[item] = a

    degree_keys = list(edge_degree.keys())
    data = set()

    for item in pairs:
        data.add(item[0] + '\t' + item[1])
        data.add(item[1] + '\t' + item[0])

    neg_edges = set()

    uniprot_ids = list(set([item[0] for item in pairs] + [item[1] for item in pairs]))
    go_similarities_pkl = os.path.join(data_dir, 'go_similarities_.pkl')

    if not os.path.exists(go_similarities_pkl):
        compute_and_store_go_similarities(uniprot_ids, go_similarities_pkl)

    with open(go_similarities_pkl, 'rb') as f:
        go_similarities = pickle.load(f)

    rejected_edges = 0
    accepted_edges = 0
    same_degree_accepted = 0
    total_attempts = 0

    with tqdm(total=len(pairs)) as pbar:
        for item in pairs:
            key = [(network.degree[item[0]] + network.degree[item[1]], 'middle'), ]
            same_degree = True

            while True:
                if key[0][0] not in edge_degree:
                    if len(key) > 1:
                        del (key[0])
                        continue
                    else:
                        break
                if not key:
                    print("Key list is empty. Exiting the while loop.")
                    break

                total_attempts += 1

                try:
                    value = edge_degree[key[0][0]]
                except IndexError:
                    print(f"IndexError: key list is {key}")
                    break

                flag = False

                for i in range(len(value)):
                    if value[i] not in data:
                        m1, m2 = value[i].split('\t')
                        go_sim = go_similarities.get((m1, m2), 1.0)

                        if go_sim is not None and go_sim <= go_similarity_threshold:
                            neg_edges.add(value[i])
                            data.add(value[i])
                            data.add(value[i].split('\t')[1] + '\t' + value[i].split('\t')[0])
                            flag = True
                            accepted_edges += 1

                            if same_degree:
                                same_degree_accepted += 1
                            break
                        else:
                            rejected_edges += 1

                if flag:
                    break

                same_degree = False
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
                del (key[0])

            pbar.update(1)

    new_neg_edges = []
    for item in neg_edges:
        m1, m2 = item.split('\t')
        new_neg_edges.append([m1, m2])

    if total_attempts > 0:
        same_degree_percentage = (same_degree_accepted / total_attempts) * 100
        print(f"same_degree_percentage: {same_degree_percentage:.2f}%")

    print(f"go_similarity_threshold: {go_similarity_threshold}")
    print(f"accepted_edges: {accepted_edges}")
    print(f"rejected_edges: {rejected_edges}")

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
    pairs = network.edges
    data_dir = 'dataset/{}'.format(args.dataset_name)
    uniprot_ids = list(set([item[0] for item in pairs] + [item[1] for item in pairs]))
    go_similarities_pkl = os.path.join(data_dir, 'go_similarities_.pkl')
    if not os.path.exists(go_similarities_pkl):
        print("Computing GO similarities...")
        compute_and_store_go_similarities(uniprot_ids, go_similarities_pkl)

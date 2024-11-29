import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
from Arguments import parser
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties

regular_font = FontProperties(family='sans-serif', style='normal', weight='regular', size=16)


def load_go_similarities(data_dir):
    go_similarities_pkl = os.path.join(data_dir, 'go_similarities_.pkl')
    with open(go_similarities_pkl, 'rb') as f:
        go_similarities = pickle.load(f)
    return go_similarities

def load_edge_degree(data_dir, validation_strategy):
    if validation_strategy == 'CV':
        edge_degree_pkl = os.path.join(data_dir, 'edge_degree_.pkl')
    else:
        edge_degree_pkl = os.path.join(data_dir, 'edge_degree_train.pkl')

    with open(edge_degree_pkl, 'rb') as f:
        edge_degree = pickle.load(f)
    return edge_degree


def convert_edge_degrees(edge_degrees):
    edge_degree_map = {}
    for degree, edge_set in edge_degrees.items():
        for edge in edge_set:
            protein1, protein2 = edge.strip().split('\t')
            edge_degree_map[(protein1, protein2)] = degree
            edge_degree_map[(protein2, protein1)] = degree  # 双向边
    return edge_degree_map

def load_edge_distances(data_dir):
    distance_file = os.path.join(data_dir, 'edge_distances.pkl')
    with open(distance_file, 'rb') as f:
        distances = pickle.load(f)
    return distances

def load_subcellular_data(data_dir):
    with open(os.path.join(data_dir, 'edge_subcellular.pkl'), 'rb') as f:
        edge_subcellular = pickle.load(f)
    return edge_subcellular

def load_predictions(predict_score_dir):
    predictions = []
    with open(predict_score_dir, 'r') as f:
        for line in f:
            data = line.strip().split(',')
            node_pair, label, pred_value = data[0], int(data[1]), float(data[2])
            node1, node2 = node_pair.split()
            predictions.append((node1, node2, label, pred_value))
    return predictions

def plot_distance_bar(data_dir, predict_score_dir, validation_strategy):

    edge_degrees = load_edge_degree(data_dir, validation_strategy)
    edge_degree_map = convert_edge_degrees(edge_degrees)
    edge_distances = load_edge_distances(data_dir)
    edge_GO_similarities = load_go_similarities(data_dir)
    predictions = load_predictions(predict_score_dir)
    edge_subcellular = load_subcellular_data(data_dir)

    distance_samples = defaultdict(list)

    for node1, node2, label, pred_value in predictions:
        distance = edge_distances.get((node1, node2), float('inf'))
        degree = edge_degree_map.get((node1, node2), 0)
        distance_samples[distance].append(degree)

    distances = []
    counts = []
    avg_degrees = []

    for distance, degrees in sorted(distance_samples.items(),
                                    key=lambda x: (x[0] == float('inf'), x[0])):
        distances.append(distance)
        counts.append(len(degrees))
        avg_degrees.append(np.mean(degrees))

    sns.set(style="white")
    plt.figure(figsize=(10, 6))

    norm = plt.Normalize(min(avg_degrees), max(avg_degrees))
    colors = plt.cm.RdBu(norm(avg_degrees))

    plt.bar(range(len(distances)), counts, color=colors)

    distance_labels = [str(int(d)) if d != float('inf') else 'inf' for d in distances]
    plt.xticks(range(len(distances)), distance_labels, fontproperties=regular_font)
    plt.yticks(fontproperties=regular_font)
    plt.xlabel('Edge Distance', fontproperties=regular_font)
    plt.ylabel('Number of Samples', fontproperties=regular_font)
    plt.title(' ', fontproperties=regular_font)

    sm = plt.cm.ScalarMappable(cmap='RdBu', norm=norm)
    sm.set_array([])
    # plt.colorbar(sm, label='Average Degree', fontsize=20)
    cbar = plt.colorbar(sm)
    cbar.set_label('Average Degree', fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()
    # output_dir = 'figures'
    # plt.savefig(os.path.join(output_dir, 'distance_bar_avg_degree.pdf'))
    plt.show()
    plot_subcellular_distribution(predictions, edge_degree_map, edge_subcellular)
    plot_go_similarity_distribution(predictions, edge_degree_map, edge_GO_similarities)


def plot_subcellular_distribution(predictions, edge_degree_map, edge_subcellular):
    pos_samples = defaultdict(lambda: [0, 0, 0])
    neg_samples = defaultdict(lambda: [0, 0, 0])

    for node1, node2, label, pred_value in predictions:
        degree = edge_degree_map.get((node1, node2), 0)
        subcellular_info = edge_subcellular.get((node1, node2), 2)

        if label == 1:
            pos_samples[degree][subcellular_info] += 1
        else:
            neg_samples[degree][subcellular_info] += 1

    pos_degree_ranges = defaultdict(lambda: [0, 0, 0])
    neg_degree_ranges = defaultdict(lambda: [0, 0, 0])

    degree_bins = list(range(0, 1000, 20))

    for degree, subcellular_counts in pos_samples.items():
        bin_idx = min(degree // 20 * 20, 980)
        pos_degree_ranges[bin_idx] = [x + y for x, y in zip(pos_degree_ranges[bin_idx], subcellular_counts)]

    for degree, subcellular_counts in neg_samples.items():
        bin_idx = min(degree // 20 * 20, 980)
        neg_degree_ranges[bin_idx] = [x + y for x, y in zip(neg_degree_ranges[bin_idx], subcellular_counts)]

    def plot_stacked_bar(degree_ranges, sample_type):
        bins = []
        unknown_counts = []
        same_loc_counts = []
        diff_loc_counts = []

        for bin_idx, subcellular_counts in sorted(degree_ranges.items()):
            bins.append(f"{bin_idx}")
            unknown_counts.append(subcellular_counts[2])
            same_loc_counts.append(subcellular_counts[1])
            diff_loc_counts.append(subcellular_counts[0])

        plt.figure(figsize=(10, 6))
        # [x + 0.5 for x in range(len(bins))]
        plt.bar([x + 0.5 for x in range(len(bins))], unknown_counts, color='white', edgecolor='black', label='Unknown (2)')
        plt.bar([x + 0.5 for x in range(len(bins))], same_loc_counts, bottom=unknown_counts, color='yellow', edgecolor='black',
                label='Same Localization (1)')
        plt.bar([x + 0.5 for x in range(len(bins))], diff_loc_counts, bottom=np.array(unknown_counts) + np.array(same_loc_counts),
                color='purple', edgecolor='black', label='Different Localization (0)')

        plt.xticks(range(0, len(bins), 2), bins[::2], fontsize=16)
        plt.yticks(fontsize=16)
        # plt.xticks(range(len(bins)), bins, rotation=45)
        plt.xlabel('Degree Range')
        plt.ylabel('Number of Samples')
        plt.title(f'{sample_type} Sample Subcellular Localization by Degree Range')

        plt.legend(handles=[
            mpatches.Patch(facecolor='white', label='Unknown (2)', edgecolor='black',),
            mpatches.Patch(facecolor='yellow', label='Same Localization (1)', edgecolor='black'),
            mpatches.Patch(facecolor='purple', label='Different Localization (0)', edgecolor='black')
        ], fontsize=16)

        plt.tight_layout()
        # output_dir = 'figures'
        # plt.savefig(os.path.join(output_dir, f'{sample_type} Sample Subcellular Localization by Degree Range.pdf'))
        plt.show()

    plot_stacked_bar(pos_degree_ranges, 'Positive')
    plot_stacked_bar(neg_degree_ranges, 'Negative')


def plot_go_similarity_distribution(predictions, edge_degree_map, edge_GO_similarities):
    pos_samples = defaultdict(list)
    neg_samples = defaultdict(list)

    for node1, node2, label, pred_value in predictions:
        go_similarity = edge_GO_similarities.get((node1, node2), None)
        degree = edge_degree_map.get((node1, node2), 0)

        if go_similarity is not None:
            if label == 1:
                pos_samples[go_similarity].append(degree)
            else:
                neg_samples[go_similarity].append(degree)

    bins = np.arange(0, 1.1, 0.1)
    bin_labels = [f'{round(b, 2)}' for b in bins[:-1]]

    def plot_bar_chart(sample_dict, sample_type):
        counts = []
        avg_degrees = []

        for i in range(len(bins) - 1):
            lower, upper = bins[i], bins[i + 1]
            degrees_in_bin = [degree for go_sim, degrees in sample_dict.items()
                              if go_sim is not None and lower <= go_sim < upper for degree in degrees]
            counts.append(len(degrees_in_bin))
            avg_degrees.append(np.mean(degrees_in_bin) if degrees_in_bin else 0)

        norm = plt.Normalize(min(avg_degrees), max(avg_degrees))
        colors = plt.cm.RdBu(norm(avg_degrees))

        plt.figure(figsize=(10, 6))
        plt.bar([x + 0.5 for x in range(len(bin_labels))], counts, color=colors, width=0.8)

        # [(x + 0.5)*0.6 for x in range(len(bin_labels))]

        plt.xticks(range(len(bin_labels)), bin_labels, fontsize=16)
        plt.yticks(fontsize=12)
        plt.xlabel('GO Similarity Range', fontsize=16)
        plt.ylabel('Number of Samples', fontsize=16)
        plt.title(f'{sample_type} Samples: GO Similarity vs Degree Distribution', fontsize=16)

        sm = plt.cm.ScalarMappable(cmap='RdBu', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Average Degree', fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        plt.tight_layout()
        # output_dir = 'figures'
        # plt.savefig(os.path.join(output_dir, f'{sample_type} Samples: Edge Degree_bar_Average GO Similarity.pdf'))
        plt.show()

    plot_bar_chart(pos_samples, 'Positive')
    plot_bar_chart(neg_samples, 'Negative')



args = parser.parse_args()
data_dir = 'dataset/{}'.format(args.dataset_name)
predict_score_dir = os.path.join(data_dir, 'predicted_{}_{}_{}_{}'.format(args.dataset_name,
                                                                          args.neg_sampling_strategy,
                                                                          args.classifier,
                                                                          args.validation_strategy))

# 调用绘图函数
plot_distance_bar(data_dir, predict_score_dir, args.validation_strategy)

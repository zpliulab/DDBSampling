import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
from Arguments import parser
import matplotlib.patches as mpatches

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
    predictions = load_predictions(predict_score_dir)
    # edge_subcellular = load_subcellular_data(data_dir)

    distance_samples = defaultdict(list)

    for node1, node2, label, pred_value in predictions:
        distance = edge_distances.get((node1, node2), float('inf'))
        degree = edge_degree_map.get((node1, node2), 0)
        distance_samples[distance].append(degree)

    distances = []
    counts = []
    avg_degrees = []

    for distance, degrees in sorted(distance_samples.items(),
                                    key=lambda x: (x[0] == float('inf'), x[0])):  # 正序排序，inf在最后
        distances.append(distance)
        counts.append(len(degrees))
        avg_degrees.append(np.mean(degrees))

    sns.set(style="white")
    plt.figure(figsize=(10, 6))


    norm = plt.Normalize(min(avg_degrees), max(avg_degrees))
    colors = plt.cm.RdBu(norm(avg_degrees))


    plt.bar(range(len(distances)), counts, color=colors)


    distance_labels = [str(int(d)) if d != float('inf') else 'inf' for d in distances]
    plt.xticks(range(len(distances)), distance_labels, fontsize=16)  # 横坐标不倾斜
    plt.yticks(fontsize=16)
    plt.xlabel('Edge Distance', fontsize=20)
    plt.ylabel('Number of Samples', fontsize=20)
    plt.title(' ', fontsize=20)

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

    plot_distance_bar_with_avg_prediction(data_dir, predict_score_dir)
    plot_degree_distribution(predictions, edge_degree_map)
    plot_negative_sample_degree_distribution(predictions, edge_degree_map, edge_distances)
    # plot_subcellular_distribution(predictions, edge_degree_map, edge_subcellular)

def plot_distance_bar_with_avg_prediction(data_dir, predict_score_dir):
    edge_distances = load_edge_distances(data_dir)
    predictions = load_predictions(predict_score_dir)

    distance_samples = defaultdict(list)

    for node1, node2, label, pred_value in predictions:
        distance = edge_distances.get((node1, node2), float('inf'))
        distance_samples[distance].append(pred_value)

    distances = []
    counts = []
    avg_predictions = []

    for distance, predictions in sorted(distance_samples.items(),
                                        key=lambda x: (x[0] == float('inf'), x[0])):
        distances.append(distance)
        counts.append(len(predictions))
        avg_predictions.append(np.mean(predictions))

    sns.set(style="white")
    plt.figure(figsize=(10, 6))

    norm = plt.Normalize(0.2, 0.8)
    colors = plt.cm.RdBu(norm(avg_predictions))

    plt.bar(range(len(distances)), counts, color=colors)

    distance_labels = [str(int(d)) if d != float('inf') else 'inf' for d in distances]
    plt.xticks(range(len(distances)), distance_labels, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Edge Distance', fontsize=20)
    plt.ylabel('Number of Samples', fontsize=20)
    plt.title(' ')

    sm = plt.cm.ScalarMappable(cmap='RdBu', norm=norm)
    sm.set_array([])
    # plt.colorbar(sm, label='Average Prediction Value', fontsize=20)
    cbar = plt.colorbar(sm)
    cbar.set_label('Average Prediction', fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()
    # output_dir = 'figures'
    # plt.savefig(os.path.join(output_dir, 'distance_bar_avg_prediction.pdf'))
    plt.show()


def plot_degree_distribution(predictions, edge_degree_map):
    pos_samples = defaultdict(list)
    neg_samples = defaultdict(list)

    for node1, node2, label, pred_value in predictions:
        degree = edge_degree_map.get((node1, node2), 0)
        if label == 1:
            pos_samples[degree].append(pred_value)
        else:
            neg_samples[degree].append(pred_value)

    pos_degree_ranges = defaultdict(list)
    neg_degree_ranges = defaultdict(list)

    degree_bins = list(range(0, 1000, 20))  # 0-20, 20-40, ..., 980+

    for degree, predictions in pos_samples.items():
        bin_idx = min(degree // 20 * 20, 980)
        pos_degree_ranges[bin_idx].extend(predictions)

    for degree, predictions in neg_samples.items():
        bin_idx = min(degree // 20 * 20, 980)
        neg_degree_ranges[bin_idx].extend(predictions)

    def get_range_stats(degree_ranges):
        bins = []
        counts = []
        avg_preds = []
        for bin_idx, preds in sorted(degree_ranges.items()):
            bins.append(f"{bin_idx}")
            # bins.append(f"{bin_idx}-{bin_idx + 19}")
            counts.append(len(preds))
            avg_preds.append(np.mean(preds) if preds else 0)
        return bins, counts, avg_preds

    pos_bins, pos_counts, pos_avg_preds = get_range_stats(pos_degree_ranges)
    neg_bins, neg_counts, neg_avg_preds = get_range_stats(neg_degree_ranges)

    plt.figure(figsize=(10, 6))

    norm = plt.Normalize(0.2, 0.8)

    colors = plt.cm.RdBu(norm(pos_avg_preds))
    # plt.bar(range(len(pos_bins)), pos_counts, color=colors)
    plt.bar([x + 0.5 for x in range(len(pos_bins))], pos_counts, color=colors)
    # [x - 0.5 for x in range(len(pos_bins))]
    # slicing
    plt.xticks(range(0, len(pos_bins), 2), pos_bins[::2], fontsize=16)
    plt.yticks(fontsize=16)

    plt.xlabel('Degree Range', fontsize=20)
    plt.ylabel('Number of Positive Samples', fontsize=20)
    plt.title(' ', fontsize=20)

    sm = plt.cm.ScalarMappable(cmap='RdBu', norm=norm)
    sm.set_array([])
    # plt.colorbar(sm, label='Average Prediction', fontsize=20)
    cbar = plt.colorbar(sm)
    cbar.set_label('Average Prediction', fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()
    # output_dir = 'figures'
    # plt.savefig(os.path.join(output_dir, 'degree_range_bar_avg_prediction_pos.pdf'))
    plt.show()

    plt.figure(figsize=(10, 6))

    colors = plt.cm.RdBu(norm(neg_avg_preds))
    # plt.bar(range(len(neg_bins)), neg_counts, color=colors)
    plt.bar([x + 0.5 for x in range(len(neg_bins))], neg_counts, color=colors)
    plt.xticks(range(0, len(neg_bins), 2), neg_bins[::2], fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Degree Range', fontsize=20)
    plt.ylabel('Number of Negative Samples', fontsize=20)
    plt.title(' ', fontsize=20)

    sm = plt.cm.ScalarMappable(cmap='RdBu', norm=norm)
    sm.set_array([])
    # plt.colorbar(sm, label='Average Prediction', fontsize=20)
    cbar = plt.colorbar(sm)
    cbar.set_label('Average Prediction', fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    plt.tight_layout()
    # output_dir = 'figures'
    # plt.savefig(os.path.join(output_dir, 'degree_range_bar_avg_prediction_neg.pdf'))
    plt.show()

def plot_negative_sample_degree_distribution(predictions, edge_degree_map, edge_distances):
    neg_samples = defaultdict(list)

    for node1, node2, label, pred_value in predictions:
        if label == 0:
            degree = edge_degree_map.get((node1, node2), 0)
            distance = edge_distances.get((node1, node2), float('inf'))
            neg_samples[degree].append(distance)

    neg_degree_ranges = defaultdict(list)
    degree_bins = list(range(0, 1000, 20))  # 0-20, 20-40, ..., 980+

    for degree, distances in neg_samples.items():
        bin_idx = min(degree // 20 * 20, 980)  # 将度数划分至区间
        neg_degree_ranges[bin_idx].extend(distances)

    def get_range_stats(degree_ranges):
        bins = []
        counts = []
        avg_distances = []
        for bin_idx, distances in sorted(degree_ranges.items()):
            valid_distances = [d for d in distances if np.isfinite(d)]
            avg_distance = np.mean(valid_distances) if valid_distances else 0
            bins.append(f"{bin_idx}")
            counts.append(len(valid_distances))
            avg_distances.append(avg_distance)
        return bins, counts, avg_distances

    neg_bins, neg_counts, neg_avg_distances = get_range_stats(neg_degree_ranges)

    plt.figure(figsize=(10, 6))

    norm = plt.Normalize(min(neg_avg_distances), max(neg_avg_distances))
    colors = plt.cm.RdBu(norm(neg_avg_distances))

    plt.bar([x + 0.5 for x in range(len(neg_bins))], neg_counts, color=colors)
    plt.xticks(range(0, len(neg_bins), 2), neg_bins[::2], fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Degree Range', fontsize=20)
    plt.ylabel('Number of Negative Samples', fontsize=20)
    plt.title(' ', fontsize=20)

    sm = plt.cm.ScalarMappable(cmap='RdBu', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Average Distance', fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()
    # output_dir = 'figures'
    # plt.savefig(os.path.join(output_dir, 'degree_range_bar_avg_distance_neg.pdf'))
    plt.show()


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
            bins.append(f"{bin_idx}-{bin_idx + 19}")
            unknown_counts.append(subcellular_counts[2])
            same_loc_counts.append(subcellular_counts[1])
            diff_loc_counts.append(subcellular_counts[0])

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(bins)), unknown_counts, color='white', edgecolor='black', label='Unknown (2)')
        plt.bar(range(len(bins)), same_loc_counts, bottom=unknown_counts, color='yellow', edgecolor='black',
                label='Same Localization (1)')
        plt.bar(range(len(bins)), diff_loc_counts, bottom=np.array(unknown_counts) + np.array(same_loc_counts),
                color='purple', edgecolor='black', label='Different Localization (0)')

        plt.xticks(range(len(bins)), bins, rotation=45)
        plt.xlabel('Degree Range')
        plt.ylabel('Number of Samples')
        plt.title(f'{sample_type} Sample Subcellular Localization by Degree Range')

        plt.legend(handles=[
            mpatches.Patch(facecolor='white', label='Unknown (2)', edgecolor='black'),
            mpatches.Patch(facecolor='yellow', label='Same Localization (1)', edgecolor='black'),
            mpatches.Patch(facecolor='purple', label='Different Localization (0)', edgecolor='black')
        ])

        plt.tight_layout()
        plt.show()

    plot_stacked_bar(pos_degree_ranges, 'Positive')
    plot_stacked_bar(neg_degree_ranges, 'Negative')

args = parser.parse_args()
data_dir = 'dataset/{}'.format(args.dataset_name)
predict_score_dir = os.path.join(data_dir, 'predicted_{}_{}_{}_{}'.format(args.dataset_name,
                                                                          args.neg_sampling_strategy,
                                                                          args.classifier,
                                                                          args.validation_strategy))

plot_distance_bar(data_dir, predict_score_dir, args.validation_strategy)

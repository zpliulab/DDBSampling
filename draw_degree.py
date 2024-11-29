from Arguments import parser
import os
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
args = parser.parse_args()

data_dir = 'dataset/{}'.format(args.dataset_name)
with open(os.path.join(data_dir, 'edge_degree_.pkl'), 'rb') as pid:
    degree2edge = pickle.load(pid)

edge2degree = {}
for degree in tqdm(degree2edge.keys()):
    for edge in degree2edge[degree]:
        edge2degree[edge] = np.log2(degree)


predict_score_dir = os.path.join(data_dir, 'predicted_{}_{}_{}_{}'.format(args.dataset_name, args.neg_sampling_strategy,
                                                            args.classifier, args.validation_strategy))
result = pd.read_csv(predict_score_dir)
result = result.to_numpy()
pos = result[result[:,1]==1]
neg = result[result[:,1]==0]
result = np.vstack([pos, neg])
degree = []
score = []
label = []
origin = pd.DataFrame()
for item in result:
    if item[0] in edge2degree.keys():
        degree.append(edge2degree[item[0]])
    else:
        degree.append(edge2degree[item[0].split('\t')[1]+'\t'+item[0].split('\t')[0]])
    label.append(item[1])
    score.append(item[2])

num = pos.shape[0]
degree_pos = degree[:num]
degree_neg = degree[num:]
score_pos = score[:num]
score_neg = score[num:]
pair_pos = result[:num, 0]
pair_neg = result[num:, 0]

print(spearmanr(degree_pos, score_pos))
print(spearmanr(degree_neg, score_neg))



pos_degree_log = []
for item in degree_pos:
    pos_degree_log.append(np.round(item, 1))
pos_degree_prob = {}
for i in range(len(degree_pos)):
    if pos_degree_log[i] not in pos_degree_prob.keys():
        pos_degree_prob[pos_degree_log[i]] = []
    pos_degree_prob[pos_degree_log[i]].append(score_pos[i])


pos_degree_prob = dict(sorted(pos_degree_prob.items(), key=lambda d : d[0]))
pos_degree_average_prob = []
pos_degree_average_prob_var = []
pos_x = []
pos_y = []
for key, value in pos_degree_prob.items():
    pos_x.append(key)
    pos_y.append(np.mean(value))
    pos_degree_average_prob.append([key, np.mean(value)])
    pos_degree_average_prob_var.append([key, np.mean(value), np.std(value)])

plt.bar(pos_x, pos_y, width=0.05)
plt.show()

neg_degree_log = []
for item in degree_neg:
    neg_degree_log.append(np.round(item, 1))
neg_degree_prob = {}
for i in range(len(degree_neg)):
    if neg_degree_log[i] not in neg_degree_prob.keys():
        neg_degree_prob[neg_degree_log[i]] = []
    neg_degree_prob[neg_degree_log[i]].append(score_neg[i])
neg_degree_prob = dict(sorted(neg_degree_prob.items(), key=lambda d : d[0]))
neg_degree_average_prob = []
neg_degree_average_prob_var = []
neg_x = []
neg_y = []
for key, value in neg_degree_prob.items():
    neg_x.append(key)
    neg_y.append(np.mean(value))
    neg_degree_average_prob.append([key, np.mean(value)])
    neg_degree_average_prob_var.append([key, np.mean(value), np.std(value)])
plt.bar(neg_x, neg_y, width=0.05)
plt.show()

pos_degree_average_prob_var = np.array(pos_degree_average_prob_var)
neg_degree_average_prob_var = np.array(neg_degree_average_prob_var)

data_pos = pd.DataFrame({'degree': pos_degree_average_prob_var[:,0],
                         'score': pos_degree_average_prob_var[:,1], 'std': pos_degree_average_prob_var[:,2]})

data_neg = pd.DataFrame({'degree': neg_degree_average_prob_var[:,0],
                         'score': neg_degree_average_prob_var[:,1], 'std': neg_degree_average_prob_var[:,2]})
data_dir = os.path.join(data_dir, 'result')
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
data_pos.to_csv(os.path.join(data_dir, '{}_{}_posi_proba.csv'.format(args.classifier,args.neg_sampling_strategy)))
data_neg.to_csv(os.path.join(data_dir, '{}_{}_nega_proba.csv'.format(args.classifier,args.neg_sampling_strategy)))

degree_pos = pd.DataFrame(np.array(degree_pos))
degree_neg = pd.DataFrame(np.array(degree_neg))
# degree_pos.to_csv(os.path.join(data_dir, '{}_{}_posi_degree.csv'.format(args.classifier, args.neg_sampling_strategy)))
# degree_neg.to_csv(os.path.join(data_dir, '{}_{}_nega_degree.csv'.format(args.classifier, args.neg_sampling_strategy)))


#Draw positive and negative degree
data_dir = 'dataset/{}'.format(args.dataset_name)
if args.neg_sampling_strategy == 'RANDOM':
    args.neg_sampling_strategy = 'random'
if args.neg_sampling_strategy == 'RANDOM_distance':
    args.neg_sampling_strategy = f'random_distance_{args.distance}'
if args.neg_sampling_strategy == 'RANDOM_GO':
    args.neg_sampling_strategy = f'random_GO_{args.GO_sim}'
if args.neg_sampling_strategy == 'RANDOM_subcellular':
    args.neg_sampling_strategy = f'random_subcellular'
if args.neg_sampling_strategy == 'DDB_GO':
    args.neg_sampling_strategy = f'DDB_GO_{args.GO_sim}'
if args.neg_sampling_strategy == 'DDB_distance':
    args.neg_sampling_strategy = f'DDB_distance_{args.distance}'
train_sample = os.path.join(data_dir, 'train_{}.txt'.format(args.neg_sampling_strategy))
print(train_sample)

with open(train_sample, 'r') as pid:
    lines = pid.readlines()
all_degree = []
for line in lines:
    line = line.strip().split()

    if line[0]+'\t'+line[1] in edge2degree.keys():
        all_degree.append(edge2degree[line[0]+'\t'+line[1]])
    else:
        all_degree.append(edge2degree[line[1]+'\t'+line[0]])
    # all_degree.append(edge2degree[line[0]+'\t'+line[1]])

data_dir=os.path.join(data_dir, 'result')

pos_degree = all_degree[:int(len(all_degree)*0.5)]
neg_degree = all_degree[int(len(all_degree)*0.5):]
pos_degree = pd.DataFrame(pos_degree)
neg_degree = pd.DataFrame(neg_degree)
pos_degree.to_csv(os.path.join(data_dir, '{}_{}_posi_degree.csv'.format(args.classifier, args.neg_sampling_strategy)), header=None, index=None)
neg_degree.to_csv(os.path.join(data_dir, '{}_{}_neg_degree.csv'.format(args.classifier, args.neg_sampling_strategy)),header=None, index=None)

print(roc_auc_score(label, score))

# print(spearmanr(pos_degree_average_prob_var[:,0], pos_degree_average_prob_var[:,1]))
# print(spearmanr(neg_degree_average_prob_var[:,0], neg_degree_average_prob_var[:,1]))
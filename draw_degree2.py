from Arguments import parser
import os
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
args = parser.parse_args()

data_dir = 'dataset/{}'.format(args.dataset_name)
with open(os.path.join(data_dir, 'edge_degree_.pkl'), 'rb') as pid:
    degree2edge = pickle.load(pid)

edge2degree = {}
for degree in tqdm(degree2edge.keys()):
    for edge in degree2edge[degree]:
        edge2degree[edge] = degree


result = pd.read_csv(os.path.join(data_dir, 'test_{}.txt'.format('random')))

degree = []
score = []
label = []
origin = pd.DataFrame()
for item in result.to_numpy():
    item = item[0].split()
    degree.append(np.log2(edge2degree[item[0] +'\t'+item[1]]))
    label.append(item[1])

num = int(len(degree)*0.5)

degree_pos = degree[:num]
degree_neg = degree[num:]
# score_pos = score[:num]
# score_neg = score[num:]
label_pos = label[:num]
label_neg = label[num:]
pair_pos = result.to_numpy()[:num,0]
pair_neg = result.to_numpy()[num:, 0]

# data_pos = pd.DataFrame({'pair':pair_pos, 'degree': degree_pos, 'label':label_pos, 'score': score_pos})
# data_neg = pd.DataFrame({'pair':pair_neg, 'degree': degree_neg, 'label':label_neg, 'score': score_neg})
# data_pos.to_csv(os.path.join(data_dir, '{}_positive.txt'.format(args.neg_sampling_strategy)))
# data_neg.to_csv(os.path.join(data_dir, '{}_negative.txt'.format(args.neg_sampling_strategy)))
# print(roc_auc_score(label, score))

# print(spearmanr(degree, score))
# print(spearmanr(degree_pos, score_pos))
# print(spearmanr(degree_neg, score_neg))
plt.violinplot(degree_pos)
plt.show()
plt.violinplot(degree_neg)
plt.show()
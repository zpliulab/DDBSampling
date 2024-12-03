# Negative sampling strategies impact the prediction of scale-free biomolecular network interactions with machine learning
## Environment
* python=3.7
* pip install rdkit==2023.3.2
* conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
* conda install scikit-learn==1.0.2
* conda install biopython==1.78
* conda install tqdm
* conda install networkx==2.6.3


## Usage for reproducing the results in paper
Model training
python reproduce.py --task ** --dataset_name ** --classifier ** --neg_sampling_strategy ** --validation_strategy **

These parameter can be specific from the following:
* --task from [DTI, RPI, PPI]
* --dataset_name: Select the dataset based on the task: For PPI task: Choose from ['HURI', 'BIOGRID', 'STRING', 'INBIOMAP'] For RPI task: Choose from ['NPINTER4', 'RAID2'] For DTI task: Choose from ['DRUGBANK', 'DRUGCENTRAL']

* --classifier [NOISE_RF, SEQ_RF, SEQ_AE]
* --neg_sampling_strategy Choose the negative sampling strategy from ['RANDOM', 'RANDOM_GO', 'RANDOM_distance', 'RANDOM_subcellular', 'DDB', 'DDB_GO', 'DDB_distance', 'DDB_subcellular']
* --validation_strategy [CV, c1c2c3] CV refers to the transductive evaluation, c1c2c3 refers to inductive evaluation.

* --distance: Set the distance limitation (applies to RANDOM_distance and DDB_distance strategies). This sets a maximum distance between two nodes in the negative sampling process (default is 3).
* --GO_sim: Set the GO similarity threshold (applies to RANDOM_GO, DDB_GO, RANDOM_subcellular, and DDB_subcellular strategies). This sets the minimum GO similarity for negative sampling (default is 0.1).
* --if_GO_Sub: Set to True to enable the calculation of GO similarity and subcellular localization. This option is only available when using the HURI, INBIOMAP, or BIOGRID datasets (default is True).
* --CV_frac: Set the fraction of the dataset to be used for training in cross-validation (default is 0.8).
* --fusion_type: Choose the fusion type for combining features: ['CAT', 'attention'] (default is CAT).
* --device: Specify the device for training (e.g., cuda:0 for GPU or cpu for CPU, default is cuda:0).
* --batch_size: Set the batch size for training (default is 32).

Example for training and validation:

For example, training and validation:
```
python reproduce.py --task PPI --dataset_name HURI --classifier SEQ_RF --neg_sampling_strategy RANDOM --validation_strategy CV
```

Visualization
After training and validation
```
python draw_degree.py --task PPI --dataset_name HURI --classifier SEQ_RF --neg_sampling_strategy RANDOM --validation_strategy CV
```

Use draw_degree.py to visualize the relationship between predicted probabilities and degree distribution for the current dataset and selected method.
Use draw_degree2.py to visualize the degree distribution of positive and negative samples in the form of a violin plot.
Use draw_test.py to visualize the relationship between predicted results, degree distribution, and shortest path distance between nodes in the test set.
Use draw_test2.py to explore the relationship between GO similarity, subcellular localization, and degree distribution in the test set. This visualization is only applicable to the HURI, INBIOMAP, or BIOGRID datasets, and is controlled by the --if_GO_Sub parameter.



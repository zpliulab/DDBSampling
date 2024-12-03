# Negative sampling strategies impact the prediction of scale-free biomolecular network interactions with machine learning

## Environment

*   python=3.7
*   pip install rdkit==2023.3.2
*   conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
*   conda install scikit-learn==1.0.2
*   conda install biopython==1.78
*   conda install tqdm
*   conda install networkx==2.6.3

## Usage for reproducing the results in paper

Model training

    python reproduce.py --task ** --dataset_name ** --classifier ** --neg_sampling_strategy ** --validation_strategy ** --distance ** --GO_sim ** --if_GO_Sub ** --CV_frac ** --fusion_type ** --device ** --batch_size **

These parameter can be specific from the following:

*   `--task`: Choose from \[DTI, RPI, PPI].

*   `--dataset_name`: Select the dataset based on the task:

    For PPI task: Choose from \['HURI', 'BIOGRID', 'STRING', 'INBIOMAP'].&#x20;

    For RPI task: Choose from \['NPINTER4', 'RAID2'] .

    For DTI task: Choose from \['DRUGBANK', 'DRUGCENTRAL'].

*   `--classifier`: Choose from \[NOISE\_RF, SEQ\_RF, SEQ\_AE].

*   `--neg_sampling_strategy`: Choose the negative sampling strategy from \['RANDOM', 'RANDOM\_GO', 'RANDOM\_distance', 'RANDOM\_subcellular', 'DDB', 'DDB\_GO', 'DDB\_distance', 'DDB\_subcellular'].

*   `--validation_strategy`: Choose from \[CV, c1c2c3]. CV refers to the transductive evaluation, c1c2c3 refers to inductive evaluation.

*   `--distance`: Set the distance limitation (applies to RANDOM\_distance and DDB\_distance strategies). This sets a maximum distance between two nodes in the negative sampling process (default is 3).

*   `--GO_sim`: Set the GO similarity threshold (applies to RANDOM\_GO and DDB\_GO strategies). This sets the minimum GO similarity for negative sampling (default is 0.1).

*   `--if_GO_Sub`: Set to True to enable the calculation of GO similarity and subcellular localization. This option is only available when using the HURI, INBIOMAP, or BIOGRID datasets (default is True).

*   `--CV_frac`: Set the fraction of the dataset to be used for training in cross-validation (default is 0.8).

*   `--fusion_type`: Choose the fusion type for combining features: \['CAT', 'attention'] (default is CAT).

*   `--device`: Specify the device for training (e.g., cuda:0 for GPU or cpu for CPU, default is cuda:0).

*   `--batch_size`: Set the batch size for training (default is 32).

Example for training and validation:

For example, training and validation:

    python reproduce.py --task PPI --dataset_name HURI --classifier SEQ_RF --neg_sampling_strategy RANDOM --validation_strategy CV --if_GO_Sub False

If you only wish to obtain the shortest distance information between two nodes in the current dataset, the GO similarity information between two nodes (proteins), or the subcellular localization information between two nodes (proteins), for use in subsequent visualization steps for the test dataset, follow the example below:

    python prepare_distance.py --dataset_name HURI 
    python prepare_GO.py --dataset_name HURI 
    python prepare_subcellular.py --dataset_name HURI 

If you want to attempt adding restrictions on the shortest distance information between two nodes, GO similarity information between two nodes (proteins), or subcellular localization information between two nodes (proteins) for negative samples based on Random Sampling and DDB Sampling, for further analysis, follow the example below:

    python reproduce.py --task PPI --dataset_name HURI --classifier SEQ_RF --neg_sampling_strategy RANDOM --validation_strategy CV --GO_sim 0.1 --if_GO_Sub True 

Visualization
After training and validation

    python draw_degree.py --task PPI --dataset_name HURI --classifier SEQ_RF --neg_sampling_strategy RANDOM --validation_strategy CV

Use draw\_degree.py to visualize the relationship between predicted probabilities and degree distribution for the current dataset and selected method.
Use draw\_degree2.py to visualize the degree distribution of positive and negative samples in the form of a violin plot.
Use draw\_test.py to visualize the relationship between predicted results, degree distribution, and shortest path distance between nodes in the test set.
Use draw\_test2.py to explore the relationship between GO similarity, subcellular localization, and degree distribution in the test set. This visualization is only applicable to the HURI, INBIOMAP, or BIOGRID datasets, and is controlled by the --if\_GO\_Sub parameter.


import argparse

parser = argparse.ArgumentParser(description="Network parameters")

# Main parameters
parser.add_argument(
    "--task", type=str,
    choices=['PPI', 'RPI', 'DTI'],
    help="For PPI task, HURI, BIOGRID, STRING and INBIOMAP can be chosen, "
         "for RPI task, NPINTER4 and RAID2 can be chosen, "
         "for DTI task, DRUGBANK and DRUGCENTRAL can be chosen.",
    default='PPI',
)

parser.add_argument(
    "--dataset_name", type=str,
    choices=['HURI', 'BIOGRID', 'STRING', 'INBIOMAP', 'NPINTER4', 'RAID2', 'DRUGBANK', 'DRUGCENTRAL'],
    default='BIOGRID'
)

parser.add_argument(
    "--classifier", type=str, help="classifier", choices=['NOISE_RF', 'SEQ_RF', 'Seq-AE'],
    default='SEQ_RF'
)

parser.add_argument(
    "--if_GO_Sub", type=bool, help="use the GO similarity and subcellular"
                                   "This option can only be True when 'dataset_name' is 'HURI','INBIOMAP' or 'BIOGRID'",
    default=True
)

parser.add_argument(
    "--neg_sampling_strategy", type=str, help="RANDOM or DDB or DDB_GO", choices=['RANDOM', 'RANDOM_GO', 'RANDOM_distance', 'RANDOM_subcellular', 'DDB', 'DDB_GO', 'DDB_distance', 'DDB_subcellular'],
    default='RANDOM_GO'
)

parser.add_argument(
    "--distance", type=int, help="distance limitation",
    default='3'
)

parser.add_argument(
    "--GO_sim", type=float, help="GO similarity limitation",
    default='0.1'
)

parser.add_argument(
    "--validation_strategy", type=str, help="Cross validation(CV) or c1c2c3 split", choices=['CV', 'c1c2c3'],
    default='CV'
)

parser.add_argument(
    "--CV_frac", type=float, help="Fraction for training dataset",
    default='0.8'
)
# CV 0.8

parser.add_argument(
    "--fusion_type", type=str, help="Fusion type", choices=['CAT', 'attention'],
    default='CAT'
)

parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="Which gpu/cpu to train on"
)

parser.add_argument(
    "--batch_size",
    type=int,
    default="32",
    help="batch size"
)
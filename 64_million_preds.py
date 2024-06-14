import pandas as pd
import matplotlib.pyplot as plt
import os
import random as rand
import pickle as pkl
from functions import *
from tensorflow import keras
from itertools import product
# -----------------------------------------------------------------------------------------
# INPUTS

# predictions run name (makes a directory for the preds)
# change this unless you want to overwrite previous predictions
pred_run_name = "240604_64_million_2_decimals"
# directory where the desired models are saved
# will average together predictions from these models
saved_model_dirs = ["./220924_select_rand_seed0_ROS", "./220924_select_rand_seed0_ROS", "./220924_select_rand_seed0_ROS"]
batch_size = 100000  # number of predictions to make in parallel


# --------------------------- --------------------------------------------------------------
output_dir = "./predictions/" + pred_run_name
if not os.path.isdir("./predictions"):
    os.mkdir("./predictions")
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)


AAs = [aa for aa in "GPAVLIMCFYWHKRQNEDST"]
positions_to_randomize = ['D1135', 'S1136', 'G1218', 'E1219', 'R1335', 'T1337']
# ['D1135', 'S1136', 'G1218', 'E1219', 'R1335', 'T1337'] choose from this list

mutants = []
mut_i = 0
file_num = 1
for AA1 in AAs:
    for AA2 in AAs:
        for AA3 in AAs:
            for AA4 in AAs:
                for AA5 in AAs:
                    for AA6 in AAs:
                        mut = AA1 + AA2 + AA3 + AA4 + AA5 + AA6
                        mut_i += 1
                        mutants.append(mut)
                        if mut_i == batch_size:
                            print("Working on batch %s of %s" % (file_num, 20 ** (len(positions_to_randomize)) // batch_size))
                            output_dfs_list = []
                            for model_i in range(len(saved_model_dirs)):
                                # read in the variables that were used during training
                                with open(saved_model_dirs[model_i] + "/data_parameters.pkl", 'rb') as read_file:
                                    vars = pkl.load(read_file)
                                run_name, model_name, three_nt_PAM, test_fraction, LOW_RATE_CONSTANT_CUTOFF, LOW_RATE_REPLACE, \
                                    HEATMAP_MAX, HEATMAP_MIN, nts, all_positions, pams = vars
                                with open(saved_model_dirs[model_i] + "/mean_std.pkl", 'rb') as read_file:
                                    DATA_MEAN, DATA_STD = pkl.load(read_file)

                                muts_df = pd.DataFrame([list(m) for m in mutants], columns=positions_to_randomize)
                                output_dfs_list.append(predict_from_saved_model(saved_model_dirs[model_i], muts_df, DATA_MEAN, DATA_STD, pams))


                            output_df = pd.concat(output_dfs_list)
                            output_df = output_df.groupby(positions_to_randomize).mean().reset_index()
                            output_df.to_csv("%s/predictions_batch_%s_of_%s.csv" % (output_dir, file_num, 20**len(positions_to_randomize)//batch_size),
                                            index=False)
                            mut_i = 0
                            mutants = []
                            file_num += 1

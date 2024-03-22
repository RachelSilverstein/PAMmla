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
pred_run_name = "240322_example"
# directory where the desired model is saved
saved_model_dir = "./220924_select_rand_seed0_ROS" # change this ti use different models

choose_muts = True
random_muts = False
positions_to_randomize = ['D1135', 'S1136', 'G1218', 'E1219', 'R1335', 'T1337'] # fill in if random_muts is True
# ['D1135', 'S1136', 'G1218', 'E1219', 'R1335', 'T1337'] choose from this list
n_samples = 10  # number of samples to take if random_muts is true.
chosen_muts_from_file = False  # ignores input muts and reads from file instead
chosen_muts_file = None # path to file that reads column of variant names in csv format
chosen_muts = ["LWKQQR", "DSGERT"] # name after amino acids at amino acid positions specified in positions_to_randomize
make_plots = True
# --------------------------- --------------------------------------------------------------
# override chosen muts parameter if reading in from file instead
if chosen_muts_from_file:
    chosen_muts = pd.read_csv(chosen_muts_file, header=None)
    print("making predictions for:")
    chosen_muts = [m for m in chosen_muts[0]]
    print(chosen_muts)

output_dir = "./predictions/" + pred_run_name
if not os.path.isdir("./predictions"):
    os.mkdir("./predictions")
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# read in the variables that were used during training
with open(saved_model_dir + "/data_parameters.pkl", 'rb') as read_file:
    vars = pkl.load(read_file)
run_name, model_name, three_nt_PAM, test_fraction, LOW_RATE_CONSTANT_CUTOFF, LOW_RATE_REPLACE, \
HEATMAP_MAX, HEATMAP_MIN, nts, all_positions, pams = vars
with open(saved_model_dir + "/mean_std.pkl", 'rb') as read_file:
    DATA_MEAN, DATA_STD = pkl.load(read_file)


AAs = [aa for aa in "GPAVLIMCFYWHKRQNEDST"]

if choose_muts: # format the variants into a data frame
    muts_df = pd.DataFrame([list(mut) for mut in chosen_muts], columns=all_positions, index=chosen_muts)
    muts_df = predict_from_saved_model(saved_model_dir, muts_df, DATA_MEAN, DATA_STD, pams)
    muts_df.to_csv(output_dir + "/predictions.csv")
elif random_muts: # make a data frame of randomized variants at the specified positions
    muts = []
    for i in range(n_samples):
        variant = []
        for pos in all_positions:
            if pos in positions_to_randomize:
                variant.append(rand.choice(AAs))
            else:
                variant.append(pos[0])
        muts.append(variant)
    names = ["".join(name) for name in muts]
    muts_df = pd.DataFrame([list(mut) for mut in muts], columns=all_positions, index=names)
    muts_df = predict_from_saved_model(saved_model_dir, muts_df, DATA_MEAN, DATA_STD, pams)
    muts_df.to_csv(output_dir + "/predictions.csv")

def plot_predictions(rates_csv, saved_model_dir, column_to_append_to_name=None, rates_already_log=True):
    # read in the variables that were used during training
    with open(saved_model_dir + "/data_parameters.pkl", 'rb') as read_file:
        vars = pkl.load(read_file)
    run_name, model_name, three_nt_PAM, test_fraction, LOW_RATE_CONSTANT_CUTOFF, LOW_RATE_REPLACE, \
        HEATMAP_MAX, HEATMAP_MIN, nts, all_positions, pams = vars

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    rates_df = pd.read_csv(rates_csv, index_col=0)
    rates_df = rates_df.reset_index()
    if not rates_already_log:
        rates_df[pams] = np.log10(rates_df[pams])
    for i in range(len(rates_df)):
        row = rates_df.iloc[i,]
        rates = row[pams]
        name = list(row[all_positions])
        name = "".join(name)
        if column_to_append_to_name is not None:
            name_prefix = row[column_to_append_to_name]
            name = name_prefix + "_" + name
        plt.figure(figsize=(5, 1.5))
        ax = plot_heatmap(rates, name, HEATMAP_MAX, HEATMAP_MIN)
        plt.savefig(output_dir + "/" + name + ".svg", format='svg', bbox_inches='tight')
        plt.close()


# plot the variants
if make_plots:
    plot_predictions(output_dir + "/predictions.csv", saved_model_dir)


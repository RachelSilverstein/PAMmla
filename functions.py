### Functions to be imported into other scripts

from classes import *
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle as pkl
from tensorflow import keras
import random
from georgiev_params import GEORGIEV_PARAMETERS




def filter_data(variants_of_interest_csv, all_rates):
    ### Filter the rate constant data to contain only the variants of interest and add the mutation profiles to the PAMDA data frame
    ### variants of interest csv -> contains only the variants that you want in the final data frame
    ### all_rates -> data frame with rates of all the variants that you will be filtering
    # only keep samples that vary at the positions of interest
    vars_df = pd.read_csv(variants_of_interest_csv)
    samples = set(vars_df["sample"])
    sel_df = all_rates.loc[all_rates["Sample"].isin(samples), :]
    sel_df = sel_df.reset_index(drop=True)
    # add the sample names and mutation profiles
    sel_df = sel_df.merge(vars_df, left_on="Sample", right_on="sample")  # check capitalization
    sel_df = sel_df.reset_index(drop=True)
    return sel_df


def add_onehot(PAMDA_df, muts, encoder=None, return_aas=False):
    ### Add onehot encoding of amino acids to a PAMDA data frame - mutates the PAMDA df in place, does not return a df
    ### if no encoder is provided, a new one will be fit
    ### return the fitted encoder to be used later
    PROTEIN_VOCAB = [aa for aa in "GPAVLIMCFYWHKRQNEDST"]
    if return_aas:
        return PROTEIN_VOCAB
    if encoder is None: # fit a new encoder
        cats = [PROTEIN_VOCAB for i in range(len(muts))]
        new_enc = OneHotEncoder(categories=cats)
        new_enc.fit(PAMDA_df[muts])
    else:
        new_enc = encoder
    muts_one_hot = new_enc.transform(PAMDA_df[muts]).toarray()
    PAMDA_df["muts_encoded"] = list(muts_one_hot)
    return new_enc


def add_onehot_plus_pairs(PAMDA_df, muts, encoder=None, return_aas=False):
    ### Add onehot encoding of amino acids AND ALL PAIRWISE COMBOS to a PAMDA data frame
    ###- mutates the PAMDA df in place, does not return a df
    ### if no encoder is provided, a new one will be fit
    ### return the fitted encoder to be used later
    PROTEIN_VOCAB = [aa for aa in "GPAVLIMCFYWHKRQNEDST"]
    PAIRWISE_VOCAB = [aa1 + aa2 for aa1 in PROTEIN_VOCAB for aa2 in PROTEIN_VOCAB]
    PAMDA_df, pairwise_colnames = add_pairwise_features(PAMDA_df, muts)
    feature_cols = muts + pairwise_colnames
    if return_aas:
        return PROTEIN_VOCAB
    categories_single = [PROTEIN_VOCAB for i in range(len(muts))]
    n = int(len(muts) * (len(muts) - 1) / 2)  # number of pairwise categories
    categories_double = [PAIRWISE_VOCAB for i in range(n)]
    all_categories = categories_single + categories_double
    if encoder == None:
        new_enc = OneHotEncoder(categories=all_categories)
        new_enc.fit(PAMDA_df[feature_cols])
    else:
        new_enc = encoder

    muts_one_hot = new_enc.transform(PAMDA_df[feature_cols]).toarray()
    PAMDA_df["muts_encoded"] = list(muts_one_hot)

    full_feature_names = get_full_feature_names(feature_cols, all_categories)

    return new_enc, full_feature_names


def get_full_feature_names(category_names, category_values):
    names = []
    for i in range(len(category_names)):
        cat = category_names[i]
        vals = category_values[i]
        for j in range(len(vals)):
            val = vals[j]
            names.append(str(cat) + "_" + str(val))
    return names

def add_pairwise_features(PAMDA_df, muts):
    """Add columns to PAMDA dataframe with all pairwise combos of amino acids at each position"""
    pairwise_colnames = []
    for i in range(len(muts)):
        pos1 = muts[i]
        for j in range(i + 1, len(muts)):
            pos2 = muts[j]
            colname = pos1 + "_" + pos2
            pairwise_colnames.append(colname)
            PAMDA_df[colname] = PAMDA_df[pos1].str.cat(PAMDA_df[pos2])
    return PAMDA_df, pairwise_colnames



def add_georgiev(PAMDA_df, muts, encoder=None, return_aas=False):
    ### Add Georgiev encoding of amino acids to a PAMDA data frame - mutates the PAMDA df in place, does not return a df
    ### returns None
    PROTEIN_VOCAB = [aa for aa in "GPAVLIMCFYWHKRQNEDST"]
    if return_aas:
        return PROTEIN_VOCAB
    if encoder is None:  # fit a new encoder
        cats = [PROTEIN_VOCAB for i in range(len(muts))]
        new_enc = GeorgievEncoder()
        new_enc.fit(cats)
    else:
        new_enc = encoder
    muts_georg = new_enc.transform(PAMDA_df[muts])
    PAMDA_df["muts_encoded"] = [item.flatten() for item in muts_georg]
    return new_enc


def normalize_and_log(data, DATA_MEAN, DATA_STD):
    data_norm = np.log2(data)
    data_norm = np.subtract(data_norm, DATA_MEAN)
    data_norm = np.divide(data_norm, DATA_STD)
    return data_norm


def unnormalize_and_exponentiate(data_norm, DATA_MEAN, DATA_STD):
    data = np.multiply(data_norm, DATA_STD)
    data = np.add(data, DATA_MEAN)
    data = np.exp2(data.astype(np.float64))
    return data


def normalize(data, DATA_MEAN, DATA_STD):
    data_norm = np.subtract(data, DATA_MEAN)
    data_norm = np.divide(data_norm, DATA_STD)
    return data_norm


def unnormalize(data_norm, DATA_MEAN, DATA_STD):
    data = np.multiply(data_norm, DATA_STD)
    data = np.add(data, DATA_MEAN)
    return data


def break_up_allowed_aas(allowed_aas_list, max_df_length):
    # recursively break up lists of allowed amino acids at each position until the combinatorial size of the resulting data frame
    # for each element is less than max_df_len
    n_rows = np.product([len(x) for x in allowed_aas_list], dtype='int64')
    if n_rows > max_df_length: # split on the first aa position where more than one aa is allowed
        divided_list_1 = []
        divided_list_2 = []
        first = True
        for item in allowed_aas_list:
            if (len(item) == 1) or not first:
                divided_list_1.append(item)
                divided_list_2.append(item)
            elif (len(item) > 1) and first:
                first = False
                half = int(len(item) / 2)
                first_half = item[: half]
                second_half = item[half:]
                divided_list_1.append(first_half)
                divided_list_2.append(second_half)
        return break_up_allowed_aas(divided_list_1, max_df_length) + break_up_allowed_aas(divided_list_2, max_df_length)
    else:
        return [allowed_aas_list]


def predict_from_saved_model(saved_model_dir, input_df, DATA_MEAN, DATA_STD, pams, PAM2AA=False):
    """ No longer mutates the input df. Outputs a copy"""
    model = keras.models.load_model(saved_model_dir + "/saved_model")
    with open(saved_model_dir + "/encoder.pkl", 'rb') as read_file:
        train_test_enc = pkl.load(read_file)
    if not PAM2AA: # regular AA to PAM model predictions
        # add one hot encoding of mutations
        muts_one_hot = train_test_enc.transform(input_df).toarray()
        # get predictions from loaded model
        preds = model.predict(muts_one_hot, batch_size=len(muts_one_hot))
        preds = unnormalize(preds, DATA_MEAN, DATA_STD)
        preds_df = pd.DataFrame(list(preds), columns=pams).round(decimals=2)
        output_df = input_df.copy()
        for pam in pams:
            output_df[pam] = list(preds_df[pam])
        output_df = output_df.round(decimals=2)
        return output_df
    else: # PAM to AA model
        preds = model.predict(input_df[pams], batch_size=len(input_df))
        preds_flattened = list(preds.reshape(len(input_df), 280))
        best_guesses = train_test_enc.inverse_transform(preds_flattened)
        best_guesses_strs = ["".join(element) for element in best_guesses]
        preds_df = input_df.copy()
        preds_df["predicted_best_AA_seq"] = best_guesses_strs
        preds_df["flattened_AA_prob_heatmap"] = preds_flattened
        return (preds_df, preds)  # return the raw preds also becuase we want to keep them in 2d heatmap form


def plot_heatmap(rates, name, HEATMAP_MAX, HEATMAP_MIN, axis=None):
    """Takes a row of a data frame containing rate constants - the names of the columns should be the PAMs
    ex. "NAAA", "NAAC", etc. """
    nts = ["A", "C", "G", "T"]
    # plot the rates given in a heatmap
    rates_array = np.zeros((4, 16))
    for i1 in range(4):
        for i2 in range(4):
            for i3 in range(4):
                    pam = "".join(["N", nts[i1], nts[i2], nts[i3]])
                    x_coord = (4*i2)+i3
                    y_coord = i1
                    rate = float(rates[pam].iloc[0])
                    rates_array[y_coord][x_coord] = rate
    first_half_pams = ["NA", "NC", "NG", "NT"]
    sec_half_pams = []
    for nt1 in nts:
        for nt2 in nts:
            sec_half_pams.append("".join([nt1, nt2]))
    if axis is None:
        ax = sns.heatmap(rates_array, cmap='Blues', vmax=HEATMAP_MAX, vmin=HEATMAP_MIN, xticklabels=sec_half_pams, yticklabels=first_half_pams)
        ax.set_title(name)
    else:
        ax = sns.heatmap(rates_array, cmap='Blues', vmax=HEATMAP_MAX, vmin=HEATMAP_MIN, ax=axis, xticklabels=sec_half_pams, yticklabels=first_half_pams)
    return ax


def plot_correlation(y_pred, y, output_path):
    r = np.corrcoef(y_pred.flatten(), (y.flatten()))
    r = str(r[0][1])[0:5]
    mse = str(round(mean_squared_error(y_true=y.flatten(), y_pred=y_pred.flatten()), 3))
    plt.figure(figsize = (5,5))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.xlabel("Actual log10(k)")
    plt.ylabel("Predicted log10(k)")
    plt.text(x=min(y.flatten()), y=max(y_pred.flatten()), s="R = " + r + " MSE = " + mse)
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()


def plot_paired_heatmaps(actual, predicted, name, HEATMAP_MAX, HEATMAP_MIN):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10,1.5))
    ax1.title.set_text(name + ' actual')
    ax2.title.set_text(name + ' predicted')
    p1 = plot_heatmap(actual, name + ' actual', HEATMAP_MAX, HEATMAP_MIN, ax1)
    p2 = plot_heatmap(predicted, name + " predicted", HEATMAP_MAX, HEATMAP_MIN, ax2)
    return p1, p2


def hamming_dist(str1, str2):
    dist = 0
    assert(len(str1) == len(str2))
    for i in range(len(str1)):
        if str2[i] != str1[i]:
            dist += 1
    return dist

def hd_to_train_set(names_to_test, train_set_names):
    """Returns a list of hamming distances from each member of the names_to_test list to the train_set_names
    names to test: list of strings of the variants you want to know the distances of
    train set names: list of strings to compare to
    returns: list of int of length len(names_to_test)
    """
    output_list = []
    for name in names_to_test:
        curr_min_hd = 100
        for train_name in train_set_names:
            hd = hamming_dist(name, train_name)
            if hd < curr_min_hd:
                curr_min_hd = hd
        output_list.append(curr_min_hd)
    return output_list


def add_max_PAM_col(dataframe, pams):
    """Assumes the dataframe has columns named after PAMs containing rates
    Adds a column to this dataframe (mutates in place) which says the identity of the most active PAM
    If the variant is not active, (Less than 10% of max axtivity on all PAMs), then max PAM is none
    This avoids over-sampling by random noise on inactive variants
    """
    PAMs_df = dataframe[pams]
    max_columns = PAMs_df.idxmax(axis=1)
    dataframe["max_PAM"] = list(max_columns)
    min_rate = PAMs_df.min().min()
    max_rate = PAMs_df.max().max()
    dif = max_rate - min_rate
    five_percent = min_rate + (0.05 * dif)
    for i in range(len(PAMs_df)):
        PAMs_row = PAMs_df.iloc[i, ]
        max_row_rate = PAMs_row.max()
        if max_row_rate < five_percent:
            dataframe.at[i, "max_PAM"] = "None"



def ros_by_column(dataframe, column):
    """Randomly oversample dataframe (where each row is a training example)
    Category in 'column' is what we will aim to balance"""
    categories = dataframe[column].unique()
    category_size_dict = {cat: sum(dataframe[column] == cat) for cat in categories}
    max_cat = max(category_size_dict.values())
    new_df = pd.DataFrame()
    for pam in category_size_dict.keys():
        slice = dataframe.loc[dataframe["max_PAM"] == pam, ]
        slice = slice.reset_index(drop=True)
        assert(len(slice) == category_size_dict[pam])
        curr_number = category_size_dict[pam]
        num_needed = max_cat - curr_number
        random_indexes = [random.randint(0, len(slice)-1) for _ in range(num_needed)]
        sampled = slice.iloc[random_indexes, :]
        slice = pd.concat([slice, sampled], axis=0)
        slice.reset_index(drop=True)
        new_df = pd.concat([new_df, slice], axis=0, ignore_index=True)
    return new_df



import os
import glob
import pandas as pd

csv_files = glob.glob("one_hidden_mlp_records/*/*trials.csv")

def make_dict(file_list, suffix_to_remove, prefix_to_remove):
    data_dict = {}
    for f in file_list:
        # Extract the middle portion: records/<key>/<file>
        key = os.path.basename(os.path.dirname(f))  # folder name
        # Remove the specified suffix from the end of the key
        if key.startswith(prefix_to_remove):
            key = key[len(prefix_to_remove):]        
        if key.endswith(suffix_to_remove):
            key = key[:-len(suffix_to_remove)]
        data_dict[key] = pd.read_csv(f)
    return data_dict

# Create dictionaries with specific suffix removal
parzen_files = [f for f in csv_files if os.path.basename(os.path.dirname(f)).endswith("_parzen")]
random_files = [f for f in csv_files if os.path.basename(os.path.dirname(f)).endswith("_random")]

parzen_dict = make_dict(parzen_files, "_parzen", "ol_")
random_dict = make_dict(random_files, "_random", "ol_")

from vizs_helpers import plot_accuracy_vs_param_panels, analyze_nhid1_vs_value, plot_nhid1_vs_value_panels

plot_accuracy_vs_param_panels(parzen_dict, random_dict,
                       acc_col="value", param_col="avg_fit_criteria", color_set="params_nhid1",
                       save_path="acc_vs_param_2x4_panels_scatter_1l.png"
                     )

summary_df = analyze_nhid1_vs_value(parzen_dict, random_dict, acc_col="value", nhid_col="params_pca_energy")
plot_nhid1_vs_value_panels(parzen_dict, random_dict, acc_col="value", nhid_col="params_pca_energy",
                            n_cols=7, figsize=(35, 5), save_path="nhid1_vs_value_panels.png")


"""
csv_files = glob.glob("three_hidden_mlp_records/*/*trials.csv")

# Create dictionaries with specific suffix removal
parzen_files = [f for f in csv_files if os.path.basename(os.path.dirname(f)).endswith("_parzen")]
random_files = [f for f in csv_files if os.path.basename(os.path.dirname(f)).endswith("_random")]
parzen_dict = make_dict(parzen_files, "_parzen", "tl_")
random_dict = make_dict(random_files, "_random", "tl_")

plot_accuracy_vs_param_panels(parzen_dict, random_dict,
                       acc_col="value", param_col="avg_fit_criteria",
                       save_path="acc_vs_param_2x4_panels_scatter_3l.png"
                     )

csv_files = glob.glob("ae_records/*/*trials.csv")

def make_dict(file_list, suffix_to_remove, prefix_to_remove):
    data_dict = {}
    for f in file_list:
        # Extract the middle portion: records/<key>/<file>
        key = os.path.basename(os.path.dirname(f))  # folder name
        # Remove the specified suffix from the end of the key
        if key.startswith(prefix_to_remove):
            key = key[len(prefix_to_remove):]        
        if key.endswith(suffix_to_remove):
            key = key[:-len(suffix_to_remove)]
        data_dict[key] = pd.read_csv(f)
    return data_dict

# Create dictionaries with specific suffix removal
parzen_files = [f for f in csv_files if os.path.basename(os.path.dirname(f)).endswith("_parzen")]
random_files = [f for f in csv_files if os.path.basename(os.path.dirname(f)).endswith("_random")]
parzen_dict = make_dict(parzen_files, "_parzen", "tl_")
random_dict = make_dict(random_files, "_random", "tl_")

plot_accuracy_vs_param_panels(parzen_dict, random_dict,
                       acc_col="user_attrs_val_recon_loss_mean", param_col="user_attrs_ed_overall_mean",
                       save_path="acc_vs_param_2x4_panels_scatter_ae.png"
                     )
"""
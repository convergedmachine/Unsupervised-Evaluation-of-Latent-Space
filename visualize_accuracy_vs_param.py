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
parzen_dict = make_dict(parzen_files, "_parzen", "tl_")
random_dict = make_dict(random_files, "_random", "tl_")

from vizs_helpers import plot_efficiency_curves_best_of_N, plot_ard_panels, plot_accuracy_vs_param_panels

plot_accuracy_vs_param_panels(parzen_dict, random_dict,
                       acc_col="value", param_col="avg_fit_criteria",
                       save_path="acc_vs_param_2x4_panels_scatter_1l.png"
                     )

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
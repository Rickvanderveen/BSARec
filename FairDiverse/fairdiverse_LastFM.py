#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd
import os
import fairdiverse
import yaml
import numpy as np
from datetime import date
import json
import os
import yaml
import os
import subprocess
import re
from pathlib import Path


# In[ ]:

# Change to the fairdiverse directory
os.chdir("fairdiverse")


# In[ ]:


# Create the dataset directory using os.makedirs instead of system command
os.makedirs("recommendation/dataset", exist_ok=True)


# In[ ]:


def print_evaluation_results(model_name, dataset_name, title, file_handle):
    if dataset_name !="":
        today = date.today()
        today_format = f"{today.year}-{today.month}-{today.day}"

        # read evaluation file
        evaluation_file = f"recommendation/log/{today_format}_{model_name}/test_result.json"

    else:
        evaluation_file = f"recommendation/log/{model_name}/test_result.json"

    with open(evaluation_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    # format metrics as table for visualisation
    table = {}
    for metric_key, value in metrics.items():
        metric, k = metric_key.split("@")
        if metric not in table:
            table[metric] = {}
        table[metric][f"@{k}"] = value

    df = pd.DataFrame(table).T
    df = df[sorted(df.columns, key=lambda x: int(x[1:]))]

    print(f"{title}")
    print(df)
    print(f"{title}", file=file_handle)
    print(df, file=file_handle)
    print("\n", file=file_handle)


# # 🧰 FairDiverse Tutorial
# ---

# ## **1. Add New Dataset 📁**
# ---

# ### Step 1: ⬇️ Download the Dataset from LastFM
# Download link: [LastFM Dataset](https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip) OR execute cell below
#
# #### What if the Dataset is Not in RecBole Format?
# Follow the steps [here](https://recbole.io/docs/user_guide/usage/running_new_dataset.html) in order to convert your data files to RecBole format which uses atomic files.

# In[ ]:
dataset_name = "LastFM"


# If folder LastFM dataset not exists
if os.path.exists('recommendation/dataset/LastFM'):
    print("LastFM dataset already exists. Skipping download and processing.")
else:
    # Download the LastFM dataset
    subprocess.run(['wget', 'https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip', '-O', 'recommendation/dataset/LastFM.zip', '-nc'], check=True)

    # Create directory for the dataset
    os.makedirs('recommendation/dataset/LastFM', exist_ok=True)

    # Extract the dataset
    subprocess.run(['unzip', '-n', 'recommendation/dataset/LastFM.zip', '-d', 'recommendation/dataset/LastFM'], check=True)

    # Remove the zip file
    subprocess.run(['rm', 'recommendation/dataset/LastFM.zip'], check=True)


    #
    # #### 🎬 MovieLens Dataset
    # In this notebook we will use the MovieLens Dataset as an example.
    #
    # GroupLens Research has collected and made available rating data sets from the MovieLens web site (https://movielens.org). This dataset describes 5-star rating and free-text tagging activity from MovieLens, a movie recommendation service.
    #
    # **Download the MovieLens dataset from RecBole: [MovieLens Dataset (RecBole processed)](https://drive.google.com/file/d/1G7_XhdSi1BhIvRETg0nN0O5tuOvbEs65/view?usp=drive_link)**
    #


    # #### **Step 2:** Place the dataset files under `~/recommendation/dataset/LastFM`
    #
    # We rename the downloaded files to match the RecBole format. The directory structure should look like this:
    #
    # ```text
    # fairdiverse
    # └── recommendation
    #         └── dataset
    #             └── LastFM
    #                 ├── artists.dat -> LastFM.item
    #                 ├── user_artists.dat -> LastFM.inter
    #                 ├── user_taggedartists-timestamps.dat -> LastFM.user

    # Rename LastFM dataset files to match RecBole format
    os.rename("recommendation/dataset/LastFM/artists.dat", "recommendation/dataset/LastFM/LastFM.item")
    os.rename("recommendation/dataset/LastFM/user_artists.dat", "recommendation/dataset/LastFM/LastFM.inter")
    os.rename("recommendation/dataset/LastFM/user_taggedartists-timestamps.dat", "recommendation/dataset/LastFM/LastFM.user")


    data_path = f"recommendation/dataset/{dataset_name}"
    os.makedirs(data_path, exist_ok=True)
    # move the dataset files in the folder


    # #### Dataset Content 📁

    # ---
    # **User Data**
    #
    # The file user_taggedartists-timestamps.dat comprising the attributes of the user tagged artists.
    #
    # Each record/line in the file has the following fields:
    #
    # - `userID`: the id of the users.
    # - `artistID`: the id of the artists.
    # - `tagID`: the id of the tags.
    # - `timestamp`: the timestamp of the user interaction.
    #
    # ---
    user_path = os.path.join(data_path, f"{dataset_name}.user")
    user_data = pd.read_csv(user_path,delimiter='\t')

    num_users = user_data["userID"].nunique()
    print(f"Data Sample")
    print(user_data.head())
    print(f"Total Users: {num_users}")


# ---
# **Item Data**
#
# The file artists.dat comprising the attributes of the artists.
#
# Each record/line in the file has the following fields:
#
# - `artistID`: the id of the artists.
# - `name`: the name of the artists.
# - `url`: the url of the artists.
# - `pictureURL`: the picture url of the artists.
#
# ---


    item_path = os.path.join(data_path, f"{dataset_name}.item")
    item_data = pd.read_csv(item_path, delimiter='\t', encoding='latin-1')
    item_data.rename(columns={'id': 'artistID'}, inplace=True)

    # Add the first tagID from the user data at the end of the item data where the artistID matches
    item_data = item_data.merge(user_data[['artistID', 'tagID']], on='artistID', how='left')

    # Convert tagID to int
    item_data['tagID'] = item_data['tagID'].fillna(0).astype(int)
    # Keep only the first tagID for each artistID
    item_data = item_data.drop_duplicates(subset=['artistID'])

    num_items = item_data["artistID"].nunique()
    print(item_data.head())
    print(f"Total Items: {num_items}")

    # save item_data
    item_data.to_csv(item_path,sep='\t', index=False)


# ---
# **Interaction Data**
#
# The file user_artists.dat comprising the weight of the user interaction with the artists.
#
# Each record/line in the file has the following fields:
#
# userID	artistID	weight
# - `userID`: the id of the users.
# - `artistID`: the id of the artists.
# - `weight`: the weight of the user interaction with the artist.
# ---
    interaction_path = os.path.join(data_path, f"{dataset_name}.inter")
    interaction_data = pd.read_csv(interaction_path,delimiter='\t')

    # remove from interaction data items which were dropped
    interaction_data = interaction_data[interaction_data['artistID'].isin(item_data['artistID'])]

    # Add the weight from the user data to the interaction data, at the end of a row if userID and artistID match
    interaction_data = interaction_data.merge(user_data, on=['userID', 'artistID'], how='left')
    # Drop tagID column
    interaction_data = interaction_data[['userID', 'artistID', 'weight', 'timestamp']]
    # Drop NaN values in the timestamp column
    interaction_data = interaction_data.dropna(subset=['timestamp'])
    # Convert timestamp to integer
    interaction_data['timestamp'] = interaction_data['timestamp'].astype(int)

    print(interaction_data.head())
    print("Number of items interacted with: ", len(interaction_data["artistID"].unique()))
    print("Number of users who performed an interaction: ", len(interaction_data["userID"].unique()))

    # Distirbution of ratings
    interaction_data.groupby("weight").size().reset_index()
    # save interaction_data
    interaction_data.to_csv(interaction_path, sep='\t', index=False)


# In[ ]:


config_data = {
    "user_id": "userID",
    "item_id": "artistID",
    "group_id": "tagID",
    "label_id": "weight",
    "timestamp": "timestamp",
    "text_id": "name",
    "label_threshold": 3,
    "item_domain": "music",


    "item_val": 5,
    "user_val": 5,
    "group_val": 5,
    "group_aggregation_threshold": 15,
    "sample_size": 1.0,
    "valid_ratio": 0.1,
    "test_ratio": 0.1,
    "reprocess": True,
    "sample_num": 350,
    "history_length": 20,
}

with open(f"./recommendation/properties/dataset/{dataset_name}.yaml", "w+") as file:
    yaml.dump(config_data, file, sort_keys=False)


# In[ ]:
# add dataset as a choice in main.py
with open("main.py", "r") as f:
    data = f.readlines()
    for index, line in enumerate(data):
        # Fint the line containing 'choices=[\"steam\", \"clueweb09\", \"compas\"' and parse its entries. Then add the current dataset name to the list. Then replace the whole line with the new list.
        param = re.match(r'(.*?choices=\[)(\"steam\", \"clueweb09\", \"compas\".*?)(\].*)', line)

        if param:
            # Convert the choices string to a list
            choices = set(param.group(2).replace('"', '').split(", "))
            choices.add(dataset_name)
            choises = ", ".join([f'"{c}"' for c in list(choices)])
            data[index] = f'    {param.group(1)}{choises}{param.group(3)}\n'
            break

# Write the modified content back to main.py
with open("main.py", "w") as f:
    f.writelines(data)


# In[ ]:
# ============================================================================
#    Base Model without in-processing (no fairness/diversity intervention)
# ============================================================================

# Experiment with the baselines models provided by FairDiverse
base_model_name = "BSARec"

config_base = {
    # ############ base model #########################
    "model": f"{base_model_name}",
    "data_type": "sequential",

    # Should preprocessing be redone (ignore cache)?
    "reprocess": True,

    # Fair-rank settings !!! eeds to be set to False for running the base model !!!
    # Set to True to apply fairness/diversity intervention. This is done below.
    "fair-rank": False,  # run fair-rank module or not

    # LLM recommendation setting !!! Don't change - needs to be set to False for running the base model !!!
    "use_llm": False,

    # Log name (results will be stored in ~/log/{log_name}/)
    "log_name": f"{base_model_name}_{dataset_name}",

    # ############# training parameters #################
    "device": "cpu",
    "epoch": 20,
    "batch_size": 64,
    "learning_rate": 0.001,

    # ############# evaluation parameters #################
    "mmf_eval_ratio": 0.5,
    "decimals": 4,
    "eval_step": 5,
    "eval_type": "ranking",
    "watch_metric": "mmf@20",
    "topk": [5, 10, 20],
    "store_scores": True,
    "fairness_metrics": ["MinMaxRatio", "MMF", "GINI", "Entropy"],
    "fairness_type": "Exposure"  # ["Exposure", "Utility"]
}

config_base_file = Path(f"./recommendation/{base_model_name}_base.yaml")
with open(config_base_file, "w") as file:
    yaml.dump(config_base, file, sort_keys=False)

# In[ ]:
# ============================================================================
#    Model with in-processing (FOCF)
# ============================================================================
inprocessing_model_name = "FOCF"

config_inproc = config_base.copy()
config_inproc["rank_model"] = inprocessing_model_name
config_inproc["fair-rank"] = True  # run fair-rank module or not
config_inproc["log_name"] = f"{base_model_name}_{inprocessing_model_name}_{dataset_name}"

config_inproc_file = Path(f"./recommendation/{base_model_name}_inprocessing.yaml")
with open(config_inproc_file, "w") as file:
    yaml.dump(config_inproc, file, sort_keys=False)

# In[ ]:
# ============================================================================
#    Post-processing model (CP-Fair)
# ============================================================================

postprocessing_model_name = "CPFair"
today = date.today()
today_format = f"{today.year}-{today.month}-{today.day}"

config_postproc = {
    "ranking_store_path": f"{today_format}_{base_model_name}_{dataset_name}",  # Path to the ranking score file (required for post-processing)

    # Change to any of the supported post-processing methods in Fairdiverse
    "model": f"{postprocessing_model_name}",
    "fair-rank": True,

    "log_name": f"{postprocessing_model_name}_{dataset_name}", # path to save the evaluation and the output

    # Evaluation parameters
    "topk": [5, 10, 20],
    "fairness_metrics": ["MinMaxRatio", "MMF", "GINI", "Entropy"],
    "fairness_type": "Exposure"  # "Exposure" computes exposure of item group; "Utility" computes score differences
}

config_postproc_file = Path(f"./recommendation/{base_model_name}_postprocessing.yaml")
with open(config_postproc_file, "w") as file:
    yaml.dump(config_postproc, file, sort_keys=False)


# In[ ]:
# ============================================================================
#    Running models
#    Setting `use_subprocess` to True will probably be slower
# ============================================================================
# Set this to True if you want to run the commands in a subprocess, otherwise it will just print the command
use_subprocess = False

# In[ ]:
base_command = [
    "python",
    "main.py",
    "--task", "recommendation",
    "--stage", "in-processing",
    "--dataset", dataset_name,
    "--train_config_file", config_base_file.name
]
if use_subprocess:
    subprocess.run(base_command, check=True)
else:
    print(" ".join(base_command))

# In[ ]:
inproc_command = [
    "python",
    "main.py",
    "--task", "recommendation",
    "--stage", "in-processing",
    "--dataset", dataset_name,
    "--train_config_file", config_inproc_file.name
]
if use_subprocess:
    subprocess.run(inproc_command, check=True)
else:
    print(" ".join(inproc_command))

# In[ ]:
postproc_command = [
    "python",
    "main.py",
    "--task", "recommendation",
    "--stage", "post-processing",
    "--dataset", dataset_name,
    "--train_config_file", config_postproc_file.name
]
if use_subprocess:
    subprocess.run(postproc_command, check=True)
else:
    print(" ".join(postproc_command))

#
# In[ ]:
# Create write file
model_path = Path(f"results/")
model_path.mkdir(parents=True, exist_ok=True)
with open(f"{model_path}/{base_model_name}_{dataset_name}.txt", "w") as file_handle:
    # evaluation results of the base model
    print_evaluation_results(config_base['log_name'], dataset_name, f"{base_model_name} base", file_handle)

    # evaluation results of in-processing model
    print_evaluation_results(config_inproc['log_name'], dataset_name, f"{base_model_name} in-processing ({inprocessing_model_name})", file_handle)

    # evaluation results of post-processing model
    print_evaluation_results(config_postproc['log_name'], dataset_name, f"{base_model_name} post-processing ({postprocessing_model_name})", file_handle)




# #### ✅ CP-Fair improves fairness and diversity metrics over the base model SASRec, with only a small drop in NDCG and utility loss.

# %%

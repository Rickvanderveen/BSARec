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


# In[ ]:

# Change to the fairdiverse directory
os.chdir("fairdiverse")


# In[ ]:


# Create the dataset directory using os.makedirs instead of system command
os.makedirs("recommendation/dataset", exist_ok=True)


# In[ ]:


def print_evaluation_results(model_name, dataset_name):
    if dataset_name !="":
        today = date.today()
        today_format = f"{today.year}-{today.month}-{today.day}"

        # read evaluation file
        evaluation_file = f"recommendation/log/{today_format}_{model_name}_{dataset_name}/test_result.json"

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

    print(df)


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
if not os.path.exists('recommendation/dataset/LastFM'):
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

# In[ ]:

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


# In[ ]:
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

# In[ ]:


    user_path = os.path.join(data_path, f"{dataset_name}.user")
    user_data = pd.read_csv(user_path,delimiter='\t')

    print(f"Data Sample")
    print(user_data.head())


    # In[ ]:


    num_users = user_data["userID"].nunique()
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

# In[ ]:


    item_path = os.path.join(data_path, f"{dataset_name}.item")
    item_data = pd.read_csv(item_path, delimiter='\t', encoding='latin-1')
    item_data.rename(columns={'id': 'artistID'}, inplace=True)

    # Add the first tagID from the user data at the end of the item data where the artistID matches
    item_data = item_data.merge(user_data[['artistID', 'tagID']], on='artistID', how='left')
    # Convert tagID to int
    item_data['tagID'] = item_data['tagID'].dropna().astype(int)

    print(item_data.head())


# In[ ]:


    num_items = item_data["artistID"].nunique()
    print(f"Total Items: {num_items}")


# In[ ]:
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

# In[ ]:
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


# In[ ]:
    print("Number of items interacted with: ", len(interaction_data["artistID"].unique()))
    print("Number of users who performed an interaction: ", len(interaction_data["userID"].unique()))


# In[ ]:
    # Distirbution of ratings
    interaction_data.groupby("weight").size().reset_index()


# In[ ]:
    # save interaction_data
    interaction_data.to_csv(interaction_path, sep='\t', index=False)


# #### **Step 3:** Create a configuration file for the dataset under `~/recommendation/properties/dataset{dataset_name}.yaml`
#
# ```yaml
# {
#     user_id: user_id:token, # column name of the user ID
#     item_id: item_id:token, # column name of the item ID, in this case we recommend movies
#     group_id: first_class:token, # column name of the groups to be considered for fairness, in this case we consider the genres of the movie
#     label_id: rating:float, # column name for the label, indicating the interest of the user in the item
#     timestamp: timestamp:float, # column name for the timestamp of when the interaction happened
#     text_id: movie_title:token_seq, # column name for the text ID of the item (e.g. movie name, book title)
#     label_threshold: 3, # if label exceed the value will be regarded as 1, otherwise, it will be accounted into 0 --> we consider a positive recommendation if a user rated a movie with a value higher than 3
#     item_domain: movie, # description of the dataset domain (e.g. movie, music, jobs etc.)
#
#    item_val: 5, # keep items which have at least this number of interactions
#    user_val: 5, # keep users who have at least this number of interactions
#    group_val: 5, # keep groups which have at least this number of interactions
#    group_aggregation_threshold: 15, ##If the number of items owned by a group is less than this value, those groups will be merged into a single group called the 'infrequent group'. For example, Fantasy, War, Musician, ... will be merged into one group called 'infrequent group', as the number of items belonging to this group is under the threshold.
#    sample_size: 1.0, ###Sample ratio of the whole dataset to form a new subset dataset for training.
#    valid_ratio: 0.1, ### Samples to be used for validation
#    test_ratio: 0.1, ### Samples to be used for test
#    reprocess: True, ##do you need to re-process the dataset according to your personalized requirements
#    sample_num: 300, # needs to be higher than the max number of positive samples per user
#    history_length: 20, # length of historical interactions of a user - [item_1, item_2, item_3, ...] to be considered
# }
# ```

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
# ## **2. Base Recommender System**
#
# ---

# To check that the set-up of the new dataset works well, let's train a base recommender system!

# #### **Step 1:** Define your training configuration file: `~/recommendation/train-base-model.yaml`
#
# You can change parameters specific to each model in the following configuration file: `recommendation/properties/models/<model_name>.yaml`
#
# ```yaml
# {
#    ############base model#########################
#    model: SASRec, # define the model to train
#    data_type: 'sequential', #[point, pair, sequential] # define the data_type needed by the model during training SASRec is a sequnetial recommender system, expecting the data_type to be 'sequential'
#    #############################################################
#
#    ##Should the preprocessing be redone based on the new parameters instead of using the cached files in ~/recommendation/process_dataset######
#    reprocess: True,
#    ###############################################
#
#   ####fair-rank model settings --> set all to False as we want to only train the base model without any fairness/diversity intervention
#    fair-rank: False, ##if you want to run a fair-rank module on the base models, you should set the value as True
#
#   # LLM recommendation setting
#    use_llm: False,
#
#   #############log name, it will store the evaluation result in ~log/your_log_name/
#    log_name: f"SASRec_{dataset_name}",
#   #################################################
#
#    ###########################training parameters################################
#    device: cpu,
#    epoch: 20,
#    batch_size: 64,
#    learning_rate: 0.001,
#    ###########################################################################
#
#
#    ###################################evaluation parameters: overwrite from ~/properties/evaluation.yaml######################################
#    mmf_eval_ratio: 0.5,
#    decimals: 4,
#    eval_step: 5,
#    eval_type: 'ranking',
#    watch_metric: 'mmf@20',
#    topk: [ 5,10,20 ], # if you choose the ranking settings, you can choose your top-k list
#    store_scores: True, #If set true, the all relevance scores will be stored in the ~/log/your_name/ for post-processing
#    fairness_metrics: ['MinMaxRatio', "MMF", "GINI", "Entropy"],
#    fairness_type: "Exposure", # ["Exposure", "Utility"], where Exposure only computes the exposure of item group while utility computes the ranking score of item groups
#    ###########################################################################
# }
# ```

# In[ ]:


# Experiment with the baselines models provided by FairDiverse
base_model_name = "SASRec"
config_base = {
    # ############ base model #########################
    "model": f"{base_model_name}",
    "data_type": "sequential",

    # Should preprocessing be redone (ignore cache)?
    "reprocess": True,

    # Fair-rank settings !!! Don't change - needs to be set to False for running the base model !!!
    "fair-rank": False,  # run fair-rank module or not
    # in-processing model to be used for ranking
    "rank_model": "FOCF",

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

with open(f"./recommendation/train-base-model.yaml", "w") as file:
    yaml.dump(config_base, file, sort_keys=False)


# #### **Step 2: Run the Base Recommender System**

# In[ ]:
subprocess.run([
    "python",
    "main.py",
    "--task", "recommendation",
    "--stage", "in-processing",
    "--dataset", dataset_name,
    "--train_config_file", "train-base-model.yaml"
], check=True)


# #### **Output files**
# ---
#
# **Processed Dataset Structure**
#
# The following files are generated during preprocessing and saved under `processed_dataset/{dataset_name}/`:
#
# ```text
# fairdiverse
# └── recommendation
#     └──processed_dataset/
#         └── {dataset_name}/
#             ├── iid2pid.json              # Mapping from item ID to provider/group ID
#             ├── iid2text.json             # Mapping from item ID to textual representation (e.g., title)
#             ├── movie_lens.test.CTR       # Test set for click-through rate (CTR) evaluation
#             ├── movie_lens.test.ranking   # Test set for ranking evaluation
#             ├── movie_lens.train          # Training set
#             ├── movie_lens.valid.CTR      # Validation set for CTR evaluation
#             ├── movie_lens.valid.ranking  # Validation set for ranking evaluation
#             └── process_config.yaml       # Configuration used during preprocessing
# ```
# **Log Output Directory Structure**
#
# After training, the following files are saved under the `log/` directory:
# ```text
# fairdiverse
# └── recommendation
#     └──log/
#         └── 2025-5-20_SASRec_{dataset_name}/
#             ├── best_model.pth         # Saved PyTorch model weights
#             ├── config.yaml            # Configuration used for training
#             ├── ranking_scores.npz     # Numpy array of ranking scores
#             └── test_result.json       # Evaluation metrics

# **Evaluation Results 📈**
#
# ---

# In[ ]:


print_evaluation_results(base_model_name, dataset_name)


# ## **3. Run Post-processing Model**
#
# ---

# #### **3.1 With Input from FairDiverse**
#
# ---
#
# Run the post-processing model on-top of the base recommender system that we have trained in Section 2.

# #### **Step 1:** Create a configuration file for running a post-processing intervention under
# You can change parameters specific to each model in the following configuration file: `recommendation/properties/models/<model_name>.yaml`
# ```yaml
# {
#    ###############the ranking score stored path for the post-processing##################
#    ranking_store_path: {dataset_name},
#    #######################################################################################
#
#    ### !!! Don't change - needs to be set to False as we don't run a post-processing intervention !!!
#    model: "CPFair",
#    log_name: f"CPFair_{dataset_name}",
#
#    #########################Evaluation parameters#########################################
#    topk: [5, 10, 20],
#    fairness_metrics: ['MinMaxRatio', "MMF", "GINI", "Entropy"],
#    fairness_type: "Exposure", # ["Exposure", "Utility"], where Exposure only computes the exposure of item group while utility computes the ranking score of item groups
#    #####################################################################################
# }
# ```

# In[ ]:


postprocessing_model_name = "CPFair"
today = date.today()
today_format = f"{today.year}-{today.month}-{today.day}"

config_model = {
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

with open(f"./recommendation/postprocessing_with_fairdiverse.yaml", "w") as file:
    yaml.dump(config_model, file, sort_keys=False)


# **Step 2: Run the post-processing model**

# In[ ]:


subprocess.run([
    "python",
    "main.py",
    "--task", "recommendation",
    "--stage", "post-processing",
    "--dataset", dataset_name,
    "--train_config_file", "postprocessing_with_fairdiverse.yaml"
], check=True)


# **Evaluation Results📈**
#
# ---

# ### NDCG as a Measure of Utility Loss
#
# Here, **Normalized Discounted Cumulative Gain (NDCG)** is used to quantify the **loss in utility** resulting from the post-processing intervention.
#
# Specifically, it compares the ranking produced by **CP-Fair** with the original ranking of the **base model** (e.g., *SASRec*).
#
# The formula is:
#
# $$
# \text{Mean\_NDCG@k} = \frac{1}{|U|} \sum_{u \in U} \frac{DCG_u}{IDCG_u}
# $$
#
# Where:
# -  *U* is the set of users,
# - **DCG** is computed based on the ranking produced by the post-processing intervention (e.g. CP-Fair),
# - **Ideal DCG** is computed based on the original ranking produced by the base model (e.g. SASRec).
#
# An NDCG closer to 1 indicates minimal loss in utility due to the intervention.

# ### Mean Utility Loss
#
# The **mean utility loss at rank k** across all users is defined as:
#
# $$
# U_{loss@k} = \frac{1}{|U|} \sum_{u \in U} \left[ \frac{1}{k} \left( \sum_{i=1}^{k} \text{score}_{base} {(u,i)} - \sum_{i=1}^{k} \text{score}_{post} {(u,i)} \right) \right]
# $$
#
# Where:
# - *U* is the set of users,
# - $ \text{score}_{base} {(u,i)} $  is the score assigned to the *i-th* item in the **base model's** top-*k* ranking for user *u*,
# - $ \text{score}_{post} {(u,i)} $ is the score of the *i-th* item in the **post-processing model's** top-*k* ranking for user *u*.
#
# This metric captures the **average per-item utility loss over all users**, reflecting how much the re-ranking procedure deviates from the base model in terms of utility.
#

# In[ ]:


# evaluation results of post-processing model
print_evaluation_results(postprocessing_model_name, dataset_name)


# In[ ]:


# evaluation results of the base model
print_evaluation_results(base_model_name, dataset_name)


# #### ✅ CP-Fair improves fairness and diversity metrics over the base model SASRec, with only a small drop in NDCG and utility loss.

# #### **3.2 Withoout Input from FairDiverse**
#
# ---
#

# To simulate a scenario where you did not use FairDiverse to generate the required files let's rename the already generated folder.

# In[ ]:


os.rename(f"recommendation/processed_dataset/{dataset_name}", f"recommendation/processed_dataset/{dataset_name}_fairdiverse")


# **Expected Data Format**
#
# If you want to use a model not supported by FairDiverse and run the evaluation metrics you need to have the following files:
#
# (1) `iid2pid.json` - Mapping from item ID to provider/group ID
#
# (2) `ranking_scores.npz` - Numpy array of ranking scores
#
# Otherwise use one of the Base Models or In-processing models supported by FairDiverse to generate those files.

# In[ ]:


# (1) Example of expected format for iid2pid.json file -- item_id:group_id
iid2pid = {"1488": "0", "42": "1", "2508": "2", "1084": "3", "1182": "0", "1468": "4", "2087": "3", "153": "0"}


# In[ ]:


# (2) Example of expected format for ranking_scores.npz file -- sparse matrix of users x items and the corresponding score

n_users = 50
n_items = 100
user_item_matrix = np.random.rand(n_users, n_items)
print(user_item_matrix)
print("Shape:", user_item_matrix.shape)


# #### **Step 1:** Create the files needed for running the post-processing method.
#
# To simulate a scenarion where you did not use FairDiverse to create the files, take the files generated by the previous base model (e.g. SASRec) and follow the steps below.

# #### **Step 2:** Place `ranking_scores.npz` under `~/recommendation/log/{dataset_name}`
#
# ```text
# fairdiverse
# └── recommendation
#     └──log/
#         └── {dataset_name}/
#             ├── ranking_scores.npz     # Numpy array of ranking scores
#
# ```
# #### **Step 3:** Place `iid2pid.json` under `~/recommendation/processed_dataset/{dataset_name}`
# ```text
# fairdiverse
# └── recommendation
#     └──processed_dataset/
#         └── {dataset_name}/
#             ├── iid2pid.json    # Mapping from item ID to provider/group ID
# ```

# #### **Step 4:** Save data configuration file `process_config.yaml` under `~/recommendation/processed_dataset/{dataset_name}`
# ```text
# fairdiverse
# └── recommendation
#     └──processed_dataset/
#         └── {dataset_name}/
#             ├── process_config.yaml
# ```

# In[ ]:


os.makedirs(f"recommendation/log/{dataset_name}", exist_ok=True)
os.makedirs(f"recommendation/processed_dataset/{dataset_name}", exist_ok=True)


# In[ ]:


# run this if you did not use FairDiverse to create the score file
num_users = 822 # n rows of the matrix
num_items = 1345 # n columns of the matrix
num_groups = 14 # this should correspond to the unique values from iid2pid.json
config_data["item_num"] = num_items
config_data["user_num"] = num_users
config_data["group_num"] = num_groups

print(config_data)
with open(f"recommendation/processed_dataset/{dataset_name}/process_config.yaml", "w") as file:
    yaml.dump(config_data, file, sort_keys=False)


# #### **Step 5:** Create a configuration file for running a post-processing intervention under
# You can change parameters specific to each model in the following configuration file: `recommendation/properties/models/<model_name>.yaml`
# ```yaml
# {
#    ###############the ranking score stored path for the post-processing##################
#    ranking_store_path: "{dataset_name}",
#    #######################################################################################
#
#    ### !!! Don't change - needs to be set to False as we don't run a post-processing intervention !!!
#    model: "CPFair",
#    log_name: "CPFair_{dataset_name}",
#
#    #########################Evaluation parameters#########################################
#    topk: [5, 10, 20],
#    fairness_metrics: ['MinMaxRatio', "MMF", "GINI", "Entropy"],
#    fairness_type: "Exposure", # ["Exposure", "Utility"], where Exposure only computes the exposure of item group while utility computes the ranking score of item groups
#    #####################################################################################
# }
# ```

# In[ ]:


postprocessing_model_name = "CPFair"

config_model = {
    "ranking_store_path": f"{dataset_name}",  # Path to the ranking score file (required for post-processing)

    # Change to any of the supported post-processing methods in Fairdiverse
    "model": f"{postprocessing_model_name}",
    "fair-rank": True,

    "log_name": f"{postprocessing_model_name}_without_fairdiverse_{dataset_name}", # path to save the evaluation and the output

    # Evaluation parameters
    "topk": [5, 10, 20],
    "fairness_metrics": ["MinMaxRatio", "MMF", "GINI", "Entropy"],
    "fairness_type": "Exposure"  # "Exposure" computes exposure of item group; "Utility" computes score differences
}

with open(f"./recommendation/postprocessing_without_fairdiverse.yaml", "w") as file:
    yaml.dump(config_model, file, sort_keys=False)


# #### **Step 6:** Run the post-processing model

# In[ ]:


subprocess.run([
    "python",
    "main.py",
    "--task", "recommendation",
    "--stage", "post-processing",
    "--dataset", dataset_name,
    "--train_config_file", "postprocessing_without_fairdiverse.yaml"
], check=True)


# ##### **Evaluation Results 📈**
#
# ---

# ### NDCG as a Measure of Utility Loss
#
# Here, **Normalized Discounted Cumulative Gain (NDCG)** is used to quantify the **loss in utility** resulting from the post-processing intervention.
#
# Specifically, it compares the ranking produced by **CP-Fair** with the original ranking of the **base model** (e.g., *SASRec*).
#
# The formula is:
#
# $$
# \text{Mean\_NDCG@k} = \frac{1}{|U|} \sum_{u \in U} \frac{DCG_u}{IDCG_u}
# $$
#
# Where:
# -  *U* is the set of users,
# - **DCG** is computed based on the ranking produced by the post-processing intervention (e.g. CP-Fair),
# - **Ideal DCG** is computed based on the original ranking produced by the base model (e.g. SASRec).
#
# An NDCG closer to 1 indicates minimal loss in utility due to the intervention.

# ### Mean Utility Loss
#
# The **mean utility loss at rank k** across all users is defined as:
#
# $$
# U_{loss@k} = \frac{1}{|U|} \sum_{u \in U} \left[ \frac{1}{k} \left( \sum_{i=1}^{k} \text{score}_{base} {(u,i)} - \sum_{i=1}^{k} \text{score}_{post} {(u,i)} \right) \right]
# $$
#
# Where:
# - *U* is the set of users,
# - $ \text{score}_{base} {(u,i)} $  is the score assigned to the *i-th* item in the **base model's** top-*k* ranking for user *u*,
# - $ \text{score}_{post} {(u,i)} $ is the score of the *i-th* item in the **post-processing model's** top-*k* ranking for user *u*.
#
# This metric captures the **average per-item utility loss over all users**, reflecting how much the re-ranking procedure deviates from the base model in terms of utility.
#

# In[ ]:


print_evaluation_results(postprocessing_model_name, dataset_name)


# In[ ]:


print_evaluation_results(base_model_name, dataset_name)


# ## **4. Run Evaluation 📈**
#
# ---
#
# If you want to use a model not supported by FairDiverse and run the evaluation metrics you need to have the following files:
#
# (1) `iid2pid.json` - Mapping from item ID to provider/group ID
#
# (2) `ranking_scores.npz` - Numpy array of ranking scores
#
# To simulate a scenario where you did not use FairDiverse to generate the required files let's rename the already generated folder for the base model, and run again only the evaluation.
#

# In[ ]:


# no need to run if you did this in Section 3.1
# os.rename("recommendation/processed_dataset/{dataset_name}", "recommendation/processed_dataset/{dataset_name}_fairdiverse")


# In[ ]:


# (1) Example of expected format for iid2pid.json file -- item_id:group_id
iid2pid = {"1488": "0", "42": "1", "2508": "2", "1084": "3", "1182": "0", "1468": "4", "2087": "3", "153": "0"}


# In[ ]:


# (2) Example of expected format for ranking_scores.npz file -- sparse matrix of users x items and the corresponding score

users = 50
items = 100
user_item_matrix = np.random.rand(users, items)
print(user_item_matrix)
print("Shape:", user_item_matrix.shape)


# #### **Step 1:** Create the files needed for running the evaluation
#
# To simulate a scenarion where you did not use FairDiverse to create the files, take the files generated by the previous base model (e.g. SASRec) and follow the steps below.
# #### **Step 2:** Place `ranking_scores.npz` under `~/recommendation/log/SASRec_{dataset_name}`
#
# ```text
# fairdiverse
# └── recommendation
#     └──log/
#         └── {dataset_name}/
#             ├── ranking_scores.npz     # Numpy array of ranking scores
#
# ```
# #### **Step 3:** Place `iid2pid.json` under `~/recommendation/processed_dataset/{dataset_name}`
# ```text
# fairdiverse
# └── recommendation
#     └──processed_dataset/
#         └── {dataset_name}/
#             ├── iid2pid.json    # Mapping from item ID to provider/group ID
# ```
#
#
# #### **Step 4:** Save data configuration file `process_config.yaml` under `~/recommendation/processed_dataset/{dataset_name}`
# ```text
# fairdiverse
# └── recommendation
#     └──processed_dataset/
#         └── {dataset_name}/
#             ├── process_config.yaml
# ```

# In[ ]:


os.makedirs(f"recommendation/log/{base_model_name}_{dataset_name}", exist_ok=True)
os.makedirs(f"recommendation/processed_dataset/{dataset_name}", exist_ok=True)


# In[ ]:


# run this if you did not use FairDiverse to create the score file
num_users = 822 # n rows of the matrix
num_items = 1345 # n columns of the matrix
num_groups = 14 # this should correspond to the unique values from iid2pid.json
config_data["item_num"] = num_items
config_data["user_num"] = num_users
config_data["group_num"] = num_groups

print(config_data)
with open(f"recommendation/processed_dataset/{dataset_name}/process_config.yaml", "w") as file:
    yaml.dump(config_data, file, sort_keys=False)


# #### **Step 5:** Create a configuration file for the evaluation
# ```yaml
# {
#    ###############the ranking score stored path for the post-processing##################
#    ranking_store_path: "SASRec_{dataset_name}",
#    #######################################################################################
#
#    ### !!! Don't change - needs to be set to False as we don't run a post-processing intervention !!!
#    model: False,
#    use_llm: False,
#    ###############eval output path##################
#    log_name: "eval_{dataset_name}",
#
#    #########################Evaluation parameters#########################################
#    topk: [5, 10, 20],
#    fairness_metrics: ['MinMaxRatio', "MMF", "GINI", "Entropy"],
#    fairness_type: "Exposure", # ["Exposure", "Utility"], where Exposure only computes the exposure of item group while utility computes the ranking score of item groups
#    #####################################################################################
# }

# In[ ]:


config_eval = {
    "ranking_store_path": f"{base_model_name}_{dataset_name}",  # Path to the ranking score file (required for post-processing)

    # Do not change — no post-processing model used, and no base model used as we want to just perform evaluation
    "fair-rank": False,
    # output file for evaluation results
    "log_name": f"eval_{base_model_name}_without_fairdiverse_{dataset_name}", # path to save the evaluation

    # Evaluation parameters
    "topk": [5, 10, 20],
    "fairness_metrics": ["MinMaxRatio", "MMF", "GINI", "Entropy"],
    "fairness_type": "Exposure"  # "Exposure" computes exposure of item group; "Utility" computes score differences
}


with open(f"./recommendation/evaluation.yaml", "w") as file:
    yaml.dump(config_eval, file, sort_keys=False)


# #### **Step 6:** Run the evaluation

# In[ ]:


subprocess.run(["python", "main.py", "--task", "recommendation", "--stage", "post-processing",
                "--dataset", dataset_name, "--train_config_file", "evaluation.yaml"], check=True)


# ##### **Evaluation Results📈**
#
# ---

# In[ ]:


print_evaluation_results(f"eval_{base_model_name}_without_fairdiverse", dataset_name)

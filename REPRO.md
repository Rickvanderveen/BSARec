# 🔁 Reproducibility Instructions

This document provides the a set of instructions to reproduce our project results.

To install the conda environment:
```bash
cd BSARec
conda env create -f bsarec_env.yaml
```

To train the BSARec model:
```bash
cd BSARec/src
python main.py \
  --data_name LastFM \
  --lr 0.001 \
  --alpha 0.9 \
  --c 3 \
  --num_attention_heads 1 \
  --train_name BSARec_LastFM \
  --model_type BSARec \
  --data_dir data/ \
  --seed 45
```
where:
- **data_name** is the name of the dataset
- **alpha**, **c** are hyperparameters specifically to BSARec. Different models have different hyperparameters. The model specific hyperparameters can be find in `src/utils.py`
- **num_attention_heads** and **lr** are hyperparameter and depend on the dataset and model. The best set of hyperparameters for a model and datset are mentioned in the report
- **train_name** is the name of the model that will be trained (name of the weights file and log file).
- **model_type** is the type of model that is used e.g. BSARec, FEARec, DuoRec, etc.
- **do_eval** sets the model in evaluation model and run only on the test set.
- **data_dir** is the location of the dataset.

To evaluate the predictions of a model:
```bash
python main.py \
  --data_name LastFM \
  --alpha 0.9 \
  --c 3 \
  --num_attention_heads 1 \
  --load_model BSARec_LastFM \
  --model_type BSARec \
  --do_eval \
  --data_dir data/
```
where:
- **data_name** is the name of the dataset
- **alpha**, **c** are hyperparameters specifically to BSARec. Different models have different hyperparameters. The model specific hyperparameters can be find in `src/utils.py`.
- **num_attention_heads** is also a hyperparameter and depends on the dataset. The best set of hyperparameters for a model and datset are mentioned in the report.
- **load_model** is the name of the trained model.
- **model_type** is the type of model that is used e.g. BSARec, FEARec, DuoRec, etc.
- **do_eval** sets the model in evaluation model and run only on the test set.
- **data_dir** is the location of the dataset.

## CPFair
To rerank using CPFair, the model ratings are required. These are generated during the evaluation of the model on the test (`<model_type>_<data_name>.npy`).

The reranking is done in the `cpfair.ipynb`

## FOCF
To use the fairness loss during training 1 flag and 2 extra arguments needs to be set.
1. `--fariness_reg`
2. `--data_maps_path` which expects the path to the data maps e.g. `data/category_maps/LastFM/artist_popularity_mapping.json`
3. `--category_map_path` which expects the path of the category path e.g. `data/self_processed/data_maps/LastFM_maps.json`

## Diversity
The diversity can be computed from the model predictions, category map and data map. These are all combined and used in the `fairness_and_diversity.ipynb`.

## Item Fairness
To get a new model ready for testing the item-fairness (this step can be skipped if you want to run the models and baselines that are mentioned in our report), you should create a "predictions_{model_name} folder in which you place a LastFM_rel.json file (as found in the other folders) and a "LastFM_pred.json" file with the predictions. If you only have a csv file rename it to this "{model_name}_predictions.csv", put it in the new predictions folder and add the model into the list of model names in csv2sjon.py. Then in the NBR-fairness repo run:
```bash
python csv2json.py
```
And the correct json file should appear in the prediction repo. 

Then, the item-fairness can be computed by running navigating into NBR-fairness/evaluation and running the following command, where model_name is the name of the model you want to test:

```bash
python model_performance.py --pred_folder ../predictions_{model_name} --method LastFM --model {model_name} 
```
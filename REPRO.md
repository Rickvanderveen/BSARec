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


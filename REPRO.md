# 🔁 Reproducibility Instructions

This document provides the full set of instructions to reproduce our project results from scratch, including data setup, environment configuration, training, and evaluation.

```
pip install FuzzyTM>=0.4.0
pip install fairdiverse
pip install recbole
git clone https://github.com/sohamchatterjee50/BSARec.git


cd /home/scur0992/BSARec/BSARec
conda env create -f bsarec_env.yaml
source activate bsarec

python /home/scur0992/BSARec/BSARec/src/main.py  --model_type FEARec --data_name LastFM --num_attention_heads 1 --train_name FEARec_LastFM --data_dir /home/scur0992/BSARec/BSARec/src/data/ (Baseline)

python /home/scur0992/BSARec/BSARec/src/main.py  --data_name LastFM --lr 0.0005 --alpha 0.7 --c 5 --num_attention_heads 1 --train_name BSARec_LastFM --data_dir /home/scur0992/BSARec/BSARec/src/data/  (Training)

python /home/scur0992/BSARec/BSARec/src/main.py  --data_name LastFM --alpha 0.7 --c 5 --num_attention_heads 1 --load_model BSARec_LastFM --do_eval --data_dir /home/scur0992/BSARec/BSARec/src/data/  (Testing)

```




---

## 🧱 Project Structure

```bash
.
├── data/                   # Contains raw and processed datasets
├── src/                    # All source code (models, training, evaluation)
├── requirements.txt        # Python dependencies
├── README.md               # README file
├── REPRO.md                # This file
├── XXXXX
├── XXXXX
```

---

## ⚙️ Environment Setup


Setup project by running the following commands:



```bash
# Example -- overwrite if needed
conda create -n XXXXX python=XXXX
conda activate XXXXX
pip install -r requirements.txt
```

---

## 📂 Download & Prepare Datasets

Place your datasets in the `XXXX/` directory.

### Example Dataset
```bash
mkdir -p data/example_dataset
cd data/example_dataset
wget xxxxx
python -m src.preprocess_example_dataset.py xxxx
cd ../..
```

---

## ⚙️ Configuration

Set your parameters in the config file before training. Example:


---

## 🚀 5. Training

### Baselines

Run the following command to train the baseline:

```bash
python XXXX
```

To perform inference:

```bash
python XXXX
```

Alternatively, execute the following slurm jobs:

```bash
sbatch job_scripts/train_xxxxx.job
sbatch job_scripts/infer_xxxxx.job
```

---

## 📈 Evaluation

After training, evaluate all models with:

```bash
python XXXX
```

---


## 📎 Misc. Notes (optional)

---

## 📦 Dependencies / References

This project repository uses the following frameworks / refers to the following papers:

- XXX
- XXX



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




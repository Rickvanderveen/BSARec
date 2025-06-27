# 📘 An Attentive Inductive Bias for Sequential Recommendation beyond the Self-Attention(BSARec)

# Description of the files:
- BSARec_LastFM_predictions.txt: This file contains the predictions from BSARec on LastFM. Each row correponds to a user and its list of recommended artist_ids/item_ids from BSARec
- data_maps.pkl: This is a dictionary which stores the 4 different maps to convert the ids from raw dataset to processed dataset and vice-versa for LastFM:
  'user2id': This map converts the user id from raw dataset to a new id in the processed data,
  'item2id': This map converts the item/artist id from raw dataset to a new id in the processed data,
  'id2user': This map is the reverse of user2id,ie- it converts the user id in the processed data to the user id in the raw/original data
  'id2item': This map is the reverse of item2id,ie- it converts the item/artist id in the processed data to the item/artist id in the raw/original data
- data_process.py: This is the process file that contains the logic to convert the raw LastFM dataset to a processed version as found in the repo for BSARec.
- main.py: This is the main file to reproduce BSARec. It contains the logic to train/test the model. In order to generate and save the predictions file, we need to add the below logic during testing/inference mode:
            scores, result_info, predictions = trainer.test(0)
            # Save predictions to file
            pred_path = os.path.join("/home/scur0992/BSARec/BSARec/output", "BSARec_LastFM" + '_predictions.txt')
            with open(pred_path, 'w') as f:
                for idx, pred in enumerate(predictions):
                    f.write(f"User {idx}: {pred.tolist()}\n")
- trainers.py: This file also needs to be changed to generate predictions. Below logic needs to  be added during eval/inference mode:
            scores, result_info = self.get_full_sort_score(epoch, answer_list, pred_list)
            return scores, result_info, pred_list
- beyond_accuracy_compute.ipynb: This notebook contains the logic to compute Entropy. Initally we need to map the predicted artists to genres and then compute top-K entropy.K is 6 in the notebook. It also contains the logic to map the artists to generes,ie- in the raw dataset, one artist can be mapped to multiple genres, but we only choose the one with the highest frequency. However, if the genre with the highest frequency is "Other", then we map the artist to the genre with the second highest frequency if present.
- artist_category_mapping.dat: Each artist id can have multiple genres/categories out of which we choose the majority one; artist_category is the final genre assigned to this artist.
## Results

| LastFM   | SASRec | BERT4Rec | DuoRec |FEARec   | BSARec   |
|----------|--------|----------|--------|---------|----------|
| HR@5     | 0.0413 |  0.0294  | 0.0431 | 0.0431  | 0.0523   |
| HR@10    | 0.0633 |  0.0459  | 0.0624 | 0.0587  | 0.0807   |
| HR@20    | 0.0927 |  0.0596  | 0.0963 | 0.0826  | 0.1174   |
| NDCG@5   | 0.0284 |  0.0198  | 0.0300 | 0.0304  | 0.0344   |
| NDCG@10  | 0.0355 |  0.0252  | 0.0361 | 0.0354  | 0.0435   |
| NDCG@20  | 0.0429 |  0.0286  | 0.0446 | 0.0414  | 0.0526   |

Average Entropy for Top-6 recommendations: MISSING

## 🧑‍💻 Team Members
- Emo Maat – emo.maat@student.uva.nl
- Fiona Nagelhout – fiona.nagelhout@student.uva.nl
- Akshay Sardjoe Misser – akshay.sardjoe.missier@student.uva.nl
- Rick van der Veen - rick.van.der.veen@student.uva.nl

## 👥 Supervising TAs
- Yuanna Liu (Main Supervisor)
- Soham (Co-supervisor)


---

## 🧾 Project Abstract
Transformer-based sequential recommendation systems have revolutionized the field, however models that are agnostic to frequency information in user histories suffer from oversmoothing problems inherent to self-attention. While BSARec claims to mitigate this limitation, its evaluations is predominantly focused on accuracy. This study firstly examines the reproducibility of the authors' claims. Secondly, the evaluation is broadened by assessing model performance on various fairness and diversity metrics and thirdly, further enhances the model by incorporating fairness-aware optimizations.
---

## 📊 Summary of Results


### Reproducibility

In this work, we aimed to verify the results of BSARec as reported by the authors, which claim that BSARec outperforms all other SotA models. This verification is done by running the model on an already tested and an unkown dataset, the LastFM and Diginetica dataset respectively. Using metrics such as the Hit Rate and Normalized Discounted Cumulative Gain, we find that the reported performance matches ours.

### Extensions

Furthermore, we explore the fairness and diversity of the recommendations proposed by BSARec, comparing them to the same models subjected to by the authors, and find that BSARec also recommends more fair and diverse items compared to other models.

Finally, we extend BSARec with in- and post-processing methods, to check whether we can improve the fairness and diverseness of BSARec even further. Using FOCF and CPFair processing methods, we show that BSARec's performance can be improved, but to which degree is dependent on the end goal.

---

## 🛠️ Task Definition
_Define the recommendation task you are solving (e.g., sequential, generative, content-based, collaborative, ranking, etc.). Clearly describe inputs and outputs._
---

## 📂 Datasets
| Dataset | # Users | # Items | # Interactions | Avg. Length | Entropy |
|---------|---------|---------|----------------|-------------|---------|
| [LastFM](https://grouplens.org/datasets/hetrec-2011/) | 1,090 | 3,646 | 52,551 | 48.2 | 7.829 |
| [Diginetica](https://github.com/RecoHut-Datasets/diginetica) | 14,828 | 9,440 | 119,918 | 8.1 | 8.849 |


## 📏 Metrics
|Metric | Description |
|-----|-------------|
| Hit Rate (HR) | measures if each recommendation has at least one item that corresponds to the ground truth in the top-k recommended items|
| Normalized Discounted Cumulative Gain (NDCG) | measures the ranking quality of the recommended items, considering both relevance and position in the recommendation list | A position-aware metric which takes into account the relevance for each item|
| Entropy | Entropy of the category distribution in the recommendation|
---

## 🔬 Baselines & Methods

_Describe each baseline, primary methods, and how they are implemented. Mention tools/frameworks used (e.g., Surprise, LightFM, RecBole, PyTorch)._
Describe each baseline
- [SASRec](https://arxiv.org/abs/1808.09781): A self-attention based sequential recommendation model.
- [BERT4Rec](https://arxiv.org/abs/1904.06690): A BERT-based sequential recommendation model that uses masked language modeling.
- [DuoRec](https://arxiv.org/abs/2110.05730): A dual encoder model that combines user and item representations for sequential recommendation.
- [FEARec](https://arxiv.org/abs/2304.09184): A frequency-aware sequential recommendation model that incorporates frequency information in user histories.
- [BSARec](https://arxiv.org/abs/2312.10325): A sequential recommendation model that uses an attentive inductive bias to mitigate oversmoothing problems in self-attention.



### 🧠 High-Level Description of Method

_Explain your approach in simple terms. Describe your model pipeline: data input → embedding/representation → prediction → ranking. Discuss design choices, such as use of embeddings, neural networks, or attention mechanisms._

The pipeline of the method in this repo is as follows:
1. **Data Input**: The model takes user-item interaction data as input, where each
2. **Embedding/Representation**: The model uses an embedding layer to convert user and item IDs into dense vectors.
3. **Prediction**: The model uses a self-attention mechanism and fast fourrier transform to capture the sequential patterns
4. **Ranking**: The model ranks the items based on their predicted scores, which are computed using a linear layer on the output of the self-attention mechanism.

### Extension locations
- **FOCF**: This is implemented in the in-processing and training fase of the model, where an additional fair-aware regularization loss is added to the training objective.
- **CPFair**: This is implemented in the post-processing phase, where the model's output is processed to yield a fair ranking of items.


## 🌱 Proposed Extensions
To create fairer sequential recommendations with BSARec, the model is subjected to two types of fairness algorithms. The first is FOCF, which is an in-processing method and acts as an additional fair-aware regularization loss during training. The second algorithm is CPFair and is a post-processing method, which is applied after the model produces its output, yielding a greedy solution to the problem of fair ranking.

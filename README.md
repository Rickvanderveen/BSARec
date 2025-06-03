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

| LastFM   | FEARec   | BSARec   |
|----------|----------|----------|
| HR@5     | 0.0303   | 0.0495   |
| HR@10    | 0.0413   | 0.0761   |
| HR@20    | 0.0615   | 0.1055   |
| NDCG@5   | 0.0204   | 0.0334   |
| NDCG@10  | 0.0241   | 0.0419   |
| NDCG@20  | 0.0291   | 0.0491   |

Average Entropy for Top-6 recommendations: 1.2449553407956822

## 🧑‍💻 Team Members
- Name 1 – email@example.com  
- Name 2 – email@example.com  
- Name 3 – email@example.com  

## 👥 Supervising TAs
- Yuanna Liu (Main Supervisor)
- Soham (Co-supervisor)


---

## 🧾 Project Abstract
_Provide a concise summary of your project, including the type of recommender system you're building, the key techniques used, and a brief two sentence summary of results._

---

## 📊 Summary of Results


### Reproducability 

_Summarize your key reproducability findings in bullet points._

### Extensions

_Summarize your key findings about the extensions you implemented in bullet points._

---

## 🛠️ Task Definition
_Define the recommendation task you are solving (e.g., sequential, generative, content-based, collaborative, ranking, etc.). Clearly describe inputs and outputs._

---

## 📂 Datasets

_Provide the following for all datasets, including the attributes you are considering to measure things like item fairness (for example)_:

- [ ] [Dataset Name](Link-to-dataset-DOI-or-URL)
  - [ ] Pre-processing: e.g., Removed items with fewer than 5 interactions, and users with fewer than 5 interactions
  - [ ] Subsets considered: e.g., Cold Start (5-10 items)
  - [ ] Dataset size: # users, # items, sparsity:
  - [ ] Attributes for user fairness (only include if used):
  - [ ] Attributes for item fairness (only include if used):
  - [ ] Attributes for group fairness (only include if used):
  - [ ] Other attributes (only include if used):

---

## 📏 Metrics

_Explain why these metrics are appropriate for your recommendation task and what they are measuring briefly._

- [ ] Metric #1
  - [ ] Description:


---

## 🔬 Baselines & Methods

_Describe each baseline, primary methods, and how they are implemented. Mention tools/frameworks used (e.g., Surprise, LightFM, RecBole, PyTorch)._
Describe each baseline
- [ ] [Baseline 1](Link-to-reference)
- [ ] [Baseline 2](Link-to-reference)



### 🧠 High-Level Description of Method

_Explain your approach in simple terms. Describe your model pipeline: data input → embedding/representation → prediction → ranking. Discuss design choices, such as use of embeddings, neural networks, or attention mechanisms._

---

## 🌱 Proposed Extensions

_List & briefly describe the extensions that you made to the original method, including extending evaluation e.g., other metrics or new datasets considered._




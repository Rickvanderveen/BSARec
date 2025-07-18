from pathlib import Path

import tqdm
import torch
import torch.nn.functional as F
import numpy as np

from torch.optim import Adam
from metrics import recall_at_k, ndcg_k
from dataset import DataMaps
from utils import load_category_map

FOCF_MULTIPLIER = 2
def focf_regularizer(scores: torch.Tensor):
    assert scores.shape[1] == 2, "This FOCF regularizer only works with 2 categories"
    gaps = torch.abs(scores[:, 0] - scores[:, 1])
    loss = F.smooth_l1_loss(gaps, torch.zeros_like(gaps), reduction="mean")
    return loss

def compute_group_scores(scores, data_maps: DataMaps, category_map: dict[str, str]):
    categories = sorted(set(category_map.values()))
    category_score = torch.zeros((len(scores), len(categories)))

    category_filter = torch.zeros((len(data_maps._id2item.keys())))
    for idx, (id, item_id) in enumerate(data_maps._id2item.items()):
        category = category_map[str(item_id)]
        category_filter[idx] = categories.index(category)

    for user_idx, user_item_scores in enumerate(scores):
        for category_idx, category in enumerate(categories):
            category_items = user_item_scores[category_filter == category_idx]
            category_score[user_idx, category_idx] += torch.sum(category_items)

    category_score = category_score / torch.sum(category_score, axis=1, keepdim=True)

    return category_score


class Trainer:
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, args, logger):
        super(Trainer, self).__init__()

        self.args = args
        self.logger = logger
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
 
        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader


        # Load the datamaps which contain the id2user, user2id, id2item, item2id
        if self.args.fairness_reg:
            self.data_maps = DataMaps.read_json(args.data_maps_path)
            self.category_map = load_category_map(args.category_map_path)

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        self.logger.info(f"Total Parameters: {sum([p.nelement() for p in self.model.parameters()])}")

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader, train=True)

    def valid(self, epoch):
        self.args.train_matrix = self.args.valid_rating_matrix
        return self.iteration(epoch, self.eval_dataloader, train=False)

    def test(self, epoch):
        self.args.train_matrix = self.args.test_rating_matrix
        return self.iteration(epoch, self.test_dataloader, train=False)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        original_state_dict = self.model.state_dict()
        self.logger.info(original_state_dict.keys())
        new_dict = torch.load(file_name, map_location=self.device) # add map location for cpu
        self.logger.info(new_dict.keys())
        for key in new_dict:
            if 'beta' in key:
                # print(key)
                # new_key = key.replace('beta', 'sqrt_beta')
                # original_state_dict[new_key] = new_dict[key]
                original_state_dict[key]=new_dict[key]
            else:
                original_state_dict[key]=new_dict[key]
        self.model.load_state_dict(original_state_dict)

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        # import pdb; pdb.set_trace()
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HR@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HR@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HR@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        self.logger.info(post_fix)

        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def iteration(self, epoch, dataloader, train=True):

        str_code = "train" if train else "test"
        # Setting the tqdm progress bar
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Mode_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        
        if train:
            self.model.train()
            rec_loss = 0.0

            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)

                user_ids, input_ids, answers, neg_answer, same_target = batch
                loss = self.model.calculate_loss(input_ids, answers, neg_answer, same_target, user_ids)

                if self.args.fairness_reg:
                    recommend_output = self.model.predict(input_ids, user_ids)
                    recommend_output = recommend_output[:, -1, :]

                    rating_pred = self.predict_full(recommend_output)
                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()

                    scores = torch.sigmoid(torch.from_numpy(rating_pred))[:, :-1]
                    group_scores = compute_group_scores(scores, self.data_maps, self.category_map)
                    fairness_loss = focf_regularizer(group_scores)
                    loss += FOCF_MULTIPLIER * fairness_loss

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                rec_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_loss": '{:.4f}'.format(rec_loss / len(rec_data_iter)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                self.logger.info(str(post_fix))

        else:
            self.model.eval()
            pred_list = None
            answer_list = None

            rating_preds_list = []

            for i, batch in rec_data_iter:
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, answers, _, _ = batch
                recommend_output = self.model.predict(input_ids, user_ids)
                recommend_output = recommend_output[:, -1, :]# 推荐的结果
                
                rating_pred = self.predict_full(recommend_output)
                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                
                try:
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                except: # bert4rec
                    rating_pred = rating_pred[:, :-1]
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0

                rating_preds_list.extend(list(rating_pred))

                # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                # argpartition time complexity O(n)  argsort O(nlogn)
                # The minus sign "-" indicates a larger value.
                ind = np.argpartition(rating_pred, -20)[:, -20:]
                # Take the corresponding values from the corresponding dimension 
                # according to the returned subscript to get the sub-table of each row of topk
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                # Sort the sub-tables in order of magnitude.
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                # retrieve the original subscript from index again
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)

            if self.args.do_eval:
                ratings_array = np.array(rating_preds_list)
                ratings_dir = Path(self.args.output_dir, "ratings")
                ratings_dir.mkdir(exist_ok=True)
                ratings_file = ratings_dir / f"{self.args.load_model}_ratings.npy"
                np.save(ratings_file, ratings_array)

            scores, result_info = self.get_full_sort_score(epoch, answer_list, pred_list)
            return scores, result_info, pred_list

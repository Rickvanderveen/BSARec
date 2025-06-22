import os
import re
import sys
sys.path.append("BSARec/src")
import torch
import numpy as np
from trainers import Trainer
from model import MODEL_DICT
from utils import set_logger
from dataset import get_seq_dic, get_dataloder, get_rating_matrix


class Args:

    def __init__(self, output_dir, model_name, data_dir, data_name, save_name):

        log = open(f"{output_dir}{model_name}.log").readlines()[0]

        namespc = re.findall(r"Namespace\((.+)\)", log)
        namespc = namespc[0].split(", ")
        namespc = {k: v for n in namespc for k, v in [n.split("=")]}
        for k, v in namespc.items():
            v = v.replace("\'", "")
            namespc[k] = v

            if re.search(r"\d\.?\d*", v):
                if v[0].isdigit(): # bandage fix for matching "BERT4Rec"
                    namespc[k] = float(v) if "." in v else int(v)
            else:
                namespc[k] = bool(v) if v in ("True", "False") else v

        self.__dict__.update(namespc)

        self.save_name = save_name
        self.data_dir = data_dir
        self.data_name = data_name
        self.output_dir = output_dir
        self.load_model = self.train_name = model_name
        self.checkpoint_path = os.path.join(self.output_dir, self.train_name + ".pt")
        self.same_target_path = os.path.join(self.data_dir, self.data_name + "_same_target.npy")


def get_relevance_scores(args):
    log_path = os.path.join(args.output_dir, args.train_name + ".log")
    logger = set_logger(log_path)

    seq_dic, max_item, num_users = get_seq_dic(args)
    args.item_size = max_item + 1
    args.num_users = num_users + 1

    train_dataloader, eval_dataloader, test_dataloader = get_dataloder(args, seq_dic)

    logger.info(str(args))
    model = MODEL_DICT[args.model_type.lower()](args=args)


    trainer = Trainer(
            model, train_dataloader, eval_dataloader, test_dataloader, args, logger
        )

    args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(
            args.data_name, seq_dic, max_item
        )

    trainer.load(args.checkpoint_path)
    logger.info(f"Load model from {args.checkpoint_path} for test!")

    device = "cpu"      # change to cuda
    relevance_scores = []
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        user_ids, input_ids, answers, _, _ = batch
        recommend_output = model.predict(input_ids, user_ids)
        recommend_output = recommend_output[:, -1, :] 
        rating_pred = trainer.predict_full(recommend_output) 
        rating_pred = rating_pred[:, :args.item_size]
        relevance_scores.append(rating_pred)

    relevance_scores = torch.vstack(relevance_scores).detach().numpy()
    np.save(f"{args.output_dir}{args.save_name}", 
            relevance_scores)
    print("\n\n")
    print(50*"=")
    print(f"saved to {args.output_dir}{args.save_name}.npy")
    return


if __name__ == "__main__":

    args = Args(
        output_dir="./synced/output/",
        model_name="BSARec_LastFM_CP",
        data_dir="./synced/data/",
        data_name="LastFM",
        save_name="bsarec_relmat",
    )
    get_relevance_scores(args)

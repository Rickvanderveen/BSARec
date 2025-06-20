import os
import sys
sys.path.append("BSARec/src")
import torch
import numpy as np
from trainers import Trainer
from model import MODEL_DICT
from utils import set_logger
from dataset import get_seq_dic, get_dataloder, get_rating_matrix


class Args:
    
    do_eval = True
    num_items = 10
    num_users = 10
    lr = 0.001
    batch_size = 256
    epochs = 200
    no_cuda = True           # set to False
    log_freq = 1
    patience = 10
    num_workers = 4
    seed = 42                # change?
    weight_decay = 0.0
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    gpu_id = 0
    variance = 5
    max_seq_length = 50
    hidden_size = 64
    num_hidden_layers = 2
    hidden_act = "gelu"
    attention_probs_dropout_prob = 0.5
    hidden_dropout_prob = 0.5  
    initializer_range = 0.02
    item_size = None
    valid_rating_matrix = None
    test_rating_matrix = None

    def __init__(self, model_type, output_dir, model_name, data_dir, data_name, 
                 save_name, num_attention_heads, alpha=None, c=None, mask_ratio=None):
        
        if model_type not in ("bsarec", "sasrec", "bert4rec", "duorec", "fearec"):
            raise Exception
        
        if model_type == "bsarec":
            if alpha == None or c == None:
                raise Exception
            self.alpha = alpha
            self.c = c
            
        if model_type == "bert4rec":
            if mask_ratio == None:
                raise Exception
            self.mask_ratio = mask_ratio

        self.save_name = save_name
        self.model_type = model_type    
        self.data_dir = data_dir
        self.data_name = data_name
        self.output_dir = output_dir
        self.load_model = self.train_name = model_name
        self.num_attention_heads = num_attention_heads

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
        model_type="bsarec",
        output_dir="./synced/output/",
        model_name="BSARec_LastFM_CP",
        data_dir="./synced/data/",
        data_name="LastFM",
        save_name="bsarec_relmat",
        num_attention_heads=4,

        alpha=0.5,        # (bsarec)
        c=7,              # (bsarec)
        # mask_ratio=...  # (bert4rec)
    )

    get_relevance_scores(args)
import argparse
import os
from pathlib import Path
import torch
import numpy as np

from model import MODEL_DICT
from trainers import Trainer
from utils import (
    EarlyStopping,
    check_path,
    set_seed,
    set_logger,
    get_local_time,
)
from dataset import get_seq_dic, get_dataloder, get_rating_matrix


PROJECT_ROOT = Path(__file__).parent


def parse_args():
    parser = argparse.ArgumentParser()

    # basic args
    parser.add_argument("--data_dir", default="./data/", type=str)
    parser.add_argument("--output_dir", default="hparam_search/", type=str)
    parser.add_argument("--data_name", required=True, type=str)
    parser.add_argument("--train_name", required=True, type=str)
    parser.add_argument("--num_items", default=10, type=int)
    parser.add_argument("--num_users", default=10, type=int)

    # train args
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate of adam")
    parser.add_argument(
        "--batch_size", default=256, type=int, help="number of batch_size"
    )
    parser.add_argument("--epochs", default=200, type=int, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", default=1, type=int, help="per epoch print res")
    parser.add_argument(
        "--patience",
        default=10,
        type=int,
        help="how long to wait after last time validation loss improved",
    )
    parser.add_argument("--num_workers", default=4, type=int)

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="weight_decay of adam"
    )
    parser.add_argument(
        "--adam_beta1", default=0.9, type=float, help="adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", default=0.999, type=float, help="adam second beta value"
    )
    parser.add_argument("--gpu_id", default="0", type=str, help="gpu_id")
    parser.add_argument("--variance", default=5, type=float)

    # model args
    parser.add_argument("--model_type", default="BSARec", type=str)
    parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument(
        "--hidden_size", default=64, type=int, help="embedding dimension"
    )
    parser.add_argument(
        "--num_hidden_layers", default=2, type=int, help="number of blocks"
    )
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--attention_probs_dropout_prob", default=0.5, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
    parser.add_argument("--initializer_range", default=0.02, type=float)

    args, _ = parser.parse_known_args()

    if args.model_type.lower() == "bsarec":
        parser.add_argument("--c", default=3, type=int)
        parser.add_argument("--alpha", default=0.9, type=float)

    elif args.model_type.lower() == "bert4rec":
        parser.add_argument("--mask_ratio", default=0.2, type=float)

    elif args.model_type.lower() == "caser":
        parser.add_argument("--nh", default=8, type=int)
        parser.add_argument("--nv", default=4, type=int)
        parser.add_argument("--reg_weight", default=1e-4, type=float)

    elif args.model_type.lower() == "duorec":
        parser.add_argument("--tau", default=1.0, type=float)
        parser.add_argument("--lmd", default=0.1, type=float)
        parser.add_argument("--lmd_sem", default=0.1, type=float)
        parser.add_argument("--ssl", default="us_x", type=str)
        parser.add_argument("--sim", default="dot", type=str)

    elif args.model_type.lower() == "fearec":
        parser.add_argument("--tau", default=1.0, type=float)
        parser.add_argument("--lmd", default=0.1, type=float)
        parser.add_argument("--lmd_sem", default=0.1, type=float)
        parser.add_argument("--ssl", default="us_x", type=str)
        parser.add_argument("--sim", default="dot", type=str)
        parser.add_argument("--spatial_ratio", default=0.1, type=float)
        parser.add_argument("--global_ratio", default=0.6, type=float)
        parser.add_argument("--fredom_type", default="us_x", type=str)
        parser.add_argument(
            "--fredom", default="True", type=str
        )  # use eval function to use as boolean

    elif args.model_type.lower() == "gru4rec":
        parser.add_argument(
            "--gru_hidden_size", default=64, type=int, help="hidden size of GRU"
        )

    return parser.parse_args()


def get_slurm_file_name() -> str:
    slurm_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
    slurm_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
    return f"{slurm_job_id}_{slurm_task_id}"


def main():
    args = parse_args()

    run_dir = PROJECT_ROOT.joinpath(args.output_dir, args.train_name, get_slurm_file_name())
    run_dir.mkdir(parents=True)

    log_path = run_dir / "out.log"
    logger = set_logger(log_path)

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    seq_dic, max_item, num_users = get_seq_dic(args)
    args.item_size = max_item + 1
    args.num_users = num_users + 1

    args.checkpoint_path = run_dir / "model.pt"
    args.same_target_path = os.path.join(
        args.data_dir, args.data_name + "_same_target.npy"
    )
    train_dataloader, eval_dataloader, test_dataloader = get_dataloder(args, seq_dic)

    logger.info(str(args))
    model = MODEL_DICT[args.model_type.lower()](args=args)
    logger.info(model)
    trainer = Trainer(
        model, train_dataloader, eval_dataloader, test_dataloader, args, logger
    )

    args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(
        args.data_name, seq_dic, max_item
    )

    early_stopping = EarlyStopping(
        args.checkpoint_path, logger=logger, patience=args.patience, verbose=True
    )
    for epoch in range(args.epochs):
        trainer.train(epoch)
        scores, _result_info, p_red_list = trainer.valid(epoch)
        # evaluate on MRR
        early_stopping(np.array(scores[-1:]), trainer.model)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

    logger.info("---------------Validation Score---------------")
    trainer.model.load_state_dict(torch.load(args.checkpoint_path))
    scores, result_info, _pred_list = trainer.valid(0)

    logger.info(args.train_name)
    logger.info(result_info)


main()

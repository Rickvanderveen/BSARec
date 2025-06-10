import csv
import os
from pathlib import Path
import torch
import numpy as np

from model import MODEL_DICT
from trainers import Trainer
from utils import EarlyStopping, check_path, set_seed, parse_args, set_logger
from dataset import get_seq_dic, get_dataloder, get_rating_matrix


def main():
    args = parse_args()

    if not args.do_eval:
        log_path = os.path.join(args.output_dir, args.train_name + ".log")
    else:
        log_path = os.path.join(args.output_dir, args.load_model + "_eval" + ".log")
    logger = set_logger(log_path)

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    seq_dic, max_item, num_users = get_seq_dic(args)
    args.item_size = max_item + 1
    args.num_users = num_users + 1

    args.checkpoint_path = os.path.join(args.output_dir, args.train_name + ".pt")
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

    if args.do_eval:
        if args.load_model is None:
            logger.error("No model input!")
            exit(0)

        # Load the model
        args.checkpoint_path = os.path.join(args.output_dir, args.load_model + ".pt")
        trainer.load(args.checkpoint_path)

        logger.info(f"Load model from {args.checkpoint_path} for test!")
        # Run the model on the test set
        scores, result_info, predictions = trainer.test(0)

        # Save predictions to file in the predictions folder
        predictions_dir = Path(args.output_dir, "predictions")
        predictions_dir.mkdir(exist_ok=True)

        pred_path = str(predictions_dir / f"{args.load_model}_predictions.csv")

        with open(pred_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["user_id", "item_id_predictions"])
            writer.writeheader()
            for idx, pred in enumerate(predictions):
                # f.write(f"User {idx}: {pred.tolist()}\n")
                writer.writerow(
                    {"user_id": idx, "item_id_predictions": ", ".join(map(str, pred))}
                )

        logger.info(f"Saved predictions in `{pred_path}`")

    else:
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

        logger.info("---------------Test Score---------------")
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info, _pred_list = trainer.test(0)

    logger.info(args.train_name)
    logger.info(result_info)


main()

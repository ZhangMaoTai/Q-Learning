import argparse
import os
import torch

from Classification.model import ClassificationModel, BiGRU, BiGRU2
from Classification.data import Model_Data
from Classification.Train import Trainer
from utils.util import seed_everything, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="hangman")

    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--print_freq", type=int, default=100)

    parser.add_argument("--train_json_path", type=str,
                        default="./words_240000.txt", required=False,
                        help="The train vocab path")
    parser.add_argument("--eval_json_path", type=str,
                        default="./words_10000.txt", required=False,
                        help="The eval vocab path")

    parser.add_argument("--save_dir", type=str,
                        default="./save_model", required=False)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    seed_everything(1234)
    log = get_logger(
        os.path.join(args.save_dir, "train.log")
    )

    train_dataset = Model_Data(json_path=args.train_json_path)
    eval_dataset = Model_Data(json_path=args.eval_json_path)

    data_train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    data_test_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
    )

    model = BiGRU2()

    trainer = Trainer(
        model=model,
        train_dataloader=data_train_loader,
        eval_dataloader=data_test_loader,

        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_warmup_steps=args.num_warmup_steps
    )

    trainer.train(
        print_freq=args.print_freq,
        save_dir=args.save_dir,
        max_norm=args.max_norm
    )

    log.info("Done train")


if __name__ == "__main__":
    main()

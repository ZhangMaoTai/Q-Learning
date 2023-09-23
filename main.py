import argparse
import os

from Model.models import Agent
from QLearn.experience import BasicBuffer
from QLearn.Trainer import QTrainer
from QLearn.environment import environment
from utils.util import seed_everything, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="hangman")

    parser.add_argument("--max_size", type=int, default=64 * 2,
                        help="The max size of the buffer")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="The batch_size of the buffer")

    parser.add_argument("--mini_epoch", type=int, default=1)
    parser.add_argument("--num_updates", type=int, default=10000)

    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.99)

    parser.add_argument("--vocab_path", type=str,
                        default="./words_240000.txt", required=False,
                        help="The train vocab path")
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

    env = environment(vocab_path=args.vocab_path)
    buffer = BasicBuffer(max_size=args.max_size, batch_size=args.batch_size)
    agent = Agent(learning_rate=args.learning_rate,
                  gamma=args.gamma,
                  tau=args.tau)

    log.info("Done init")

    trainer = QTrainer(
        env=env,
        agent=agent,
        replay_buffer=buffer,
        mini_epoch=args.mini_epoch
    )

    trainer.train(
        num_updates=args.num_updates,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()

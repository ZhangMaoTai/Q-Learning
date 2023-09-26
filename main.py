import argparse
import os

from Model.models import Agent
from QLearn.experience import BasicBuffer
from QLearn.Trainer import QTrainer
from QLearn.environment import environment
from QLearn.eval import Evaler
from utils.util import seed_everything, get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="hangman")

    parser.add_argument("--max_size", type=int, default=64 * 2,
                        help="The max size of the buffer")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="The batch_size of the buffer")

    parser.add_argument("--mini_epoch", type=int, default=1)
    parser.add_argument("--num_updates", type=int, default=10000)
    parser.add_argument("--per_updates_eval", type=int, default=1000)

    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.99)

    parser.add_argument("--train_vocab_path", type=str,
                        default="./words_240000.txt", required=False,
                        help="The train vocab path")
    parser.add_argument("--eval_vocab_path", type=str,
                        default="./words_10000.txt", required=False,
                        help="The eval vocab path")

    parser.add_argument("--save_dir", type=str,
                        default="./save_model", required=False)
    parser.add_argument("--load_state_path", type=str,
                        default=None, required=False)


    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    seed_everything(1234)
    log = get_logger(
        os.path.join(args.save_dir, "train.log")
    )

    train_env = environment(vocab_path=args.train_vocab_path)
    eval_env = environment(vocab_path=args.eval_vocab_path)
    buffer = BasicBuffer(max_size=args.max_size, batch_size=args.batch_size)

    agent = Agent(learning_rate=args.learning_rate,
                  gamma=args.gamma,
                  tau=args.tau,
                  num_warmup_steps=args.num_warmup_steps,
                  num_training_steps=args.num_updates * args.mini_epoch * (args.max_size / args.batch_size),
                  load_state_path=args.load_state_path
                  )

    evaler = Evaler(agent=agent, env=eval_env)

    log.info("Done init")

    trainer = QTrainer(
        train_env=train_env,
        evaler=evaler,
        agent=agent,
        replay_buffer=buffer,
        mini_epoch=args.mini_epoch,
        per_updates_eval=args.per_updates_eval,
        log=log
    )

    trainer.train(
        num_updates=args.num_updates,
        save_dir=args.save_dir,
        max_norm=args.max_norm
    )

    log.info("Done train")


if __name__ == "__main__":
    main()

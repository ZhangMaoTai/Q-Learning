import os
import time
import wandb

import numpy as np
import torch
import random
from logging import getLogger, INFO, WARN, StreamHandler, FileHandler, Formatter

from torch.utils.tensorboard import SummaryWriter


def wandb_start(is_main):
    if is_main:
        exp_name = os.path.basename(__file__).rstrip(".py")
        run_name = f"{exp_name}__{int(time.time())}"
        wandb.init(
            project="hangman",
            entity=None,
            sync_tensorboard=True,
            # config=vars(args),
            name=run_name,
            # monitor_gym=True,
            save_code=True,
        )
        writer = SummaryWriter(f"runs/{run_name}")
        return writer
    else:
        return None


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_logger(filename, rank=0):
    save_file = filename

    logger = getLogger("Q-learning")
    logger.setLevel(INFO if rank in [-1, 0] else WARN)      # rank 0/1的时候才是INFO

    handler1 = StreamHandler()
    # handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=save_file)
    # handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)

    init_log(save_file, logger)
    return logger


def init_log(log_file_path, log):
    '''Clean log'''
    if os.path.exists(log_file_path):
        with open(log_file_path, "r+") as f:
            f.seek(0)
            f.truncate()  # 清空文件
            log.info("init_log")
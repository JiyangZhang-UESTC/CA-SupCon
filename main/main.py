import _init_paths
from utils import seed_everything, create_parser, create_logger
from dataprep import *
from dataloader import load_data
from core import *

import yaml
import os
import numpy as np


def main():
    args = create_parser()  # argument from command line
    
    cfg_name = "{}.yaml".format(args.cfg_name)
    cfg_path = os.path.join("configs", args.dataset, args.exp_name, cfg_name)

    # load configuration file
    with open(cfg_path) as f:
        config = yaml.load(f, Loader = yaml.FullLoader)

    logger = create_logger(config)  # create logger

    seed_everything(logger, config["SEED"])  # fix random seed

    splits = config["DATASET"]["SPLITS"]        

    os.environ['CUDA_VISIBLE_DEVICES'] = config["TRAINING_OPT"]["CUDA_VISIBLE_DEVICES"]
    logger.info("===> CUDA_VISIBLE_DEVICES: {}".format(config["TRAINING_OPT"]["CUDA_VISIBLE_DEVICES"]))

    dataloader_dict = {split: load_data(config, logger, phase = split) for split in splits}

    model = Model_SupCon(config, dataloader_dict, logger)
    if (config["MODE"] in ["train_supcon", "train_linear", "fine_tune"]):
        model.train()
    elif (config["MODE"] == "test"):
        model.eval_ce("test", True)
        
    print("\ndone!")


if __name__ == '__main__':
    main()

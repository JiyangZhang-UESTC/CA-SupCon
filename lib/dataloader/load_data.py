from torch.utils import data
from dataset import *
from sampler import ClassAwareSampler

from torch.utils.data import DataLoader
import os

def load_data(cfg, logger, phase):
    # get data save dir
    if (cfg["DATASET"]["DATA_ROOT"] is not None):
        data_root = cfg["DATASET"]["DATA_ROOT"]
    else:
        data_root = os.path.join(cfg["OUTPUT_DIR"], cfg["DATASET_NAME"], cfg["DATASET_DESC"])
        if (cfg["IMBALANCED"]):
            data_root = os.path.join(data_root, "imbalanced", cfg["IMB_DESC"], "data")
        else:
            data_root = os.path.join(data_root, "balanced", "data")
    
    if "cwru" in cfg["DATASET_NAME"]:
        data_txt = "{}_set.txt".format(phase)
    elif "te" in cfg["DATASET_NAME"]:
        data_txt = "{}_set.npy".format(phase)
    
    logger.info("------------------{} Dataset is construced as follows---------------".format(phase))
    transforms = cfg["DATASET"]["TRANSFORMS"]
    dataset = eval(cfg["DATASET"]["DATASET_CLASS"])(data_root, data_txt, transforms, logger, phase) # get pytorch Dataset
    
    logger.info("--------------------{} DataLoader is construced as follows-------------------".format(phase))
    sampler_class = cfg["DATALOADER"]["SAMPLER"]["SAMPLER_CLASS"]  # sampler class name
    num_samples_cls = cfg["DATALOADER"]["SAMPLER"]["NUM_SAMPLER_CLS"]
    batch_size = cfg["DATALOADER"]["BATCH_SIZE"]
    shuffle = cfg["DATALOADER"]["TRAIN_LOADER_SHUFFLE"]
    drop_last = cfg["DATALOADER"]["DROP_LAST"]
    num_workers = cfg["DATALOADER"]["NUM_WORKERS"]

    if phase == "train" and sampler_class is not None:
        logger.info("===> using a custom sampler: {}.".format(sampler_class))
        sampler = eval(sampler_class)(dataset=dataset, num_samples_cls=num_samples_cls)
        logger.info("do shuffle: False\n")
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, 
                          sampler=sampler, drop_last=drop_last, num_workers=num_workers)
    
    elif phase == "train" and sampler_class is None:
        logger.info("===> using default sampler.")
        logger.info("===> do shuffle: {}\n".format(shuffle))
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                          drop_last=drop_last, num_workers=num_workers)

    else:
        logger.info("===> do shuffle: False\n")
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                          drop_last=False, num_workers=num_workers)

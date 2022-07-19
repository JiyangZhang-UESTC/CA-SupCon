import torch.nn as nn

def create_ce_loss(logger, device):
    logger.info("===> create cross-entropy loss")

    return nn.CrossEntropyLoss().to(device)

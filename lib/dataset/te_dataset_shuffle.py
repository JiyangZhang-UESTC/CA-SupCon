import torch
from torch.utils.data import Dataset
import os
import numpy as np

from utils import *


class TEDatasetShuffle(Dataset):
    def __init__(self, data_root, data_txt, transforms, logger, phase):
        self.phase = phase

        self.data_path = os.path.join(data_root, data_txt)
        self.data = np.load(self.data_path, allow_pickle=True)

        self.sequence = torch.from_numpy(self.data.item().get("sequence"))
        self.label = torch.from_numpy(self.data.item().get("label")).long()

        self.sequence = self.sequence.permute(0, 2, 1)

        self.transforms = transforms
        if self.phase == "train" and self.transforms["USE_TRANSFORMS"]:
            self.transforms_list = []
            self.transform_type = self.transforms["TRANSFORM_TYPE"]
            for targets_i in range(len(self.transform_type)):
                self.transform_list = []
                for target in self.transform_type[targets_i]:
                    self.transform_list.append(eval(target)(self.transforms["PARAM_DICT"][targets_i][target]))
                self.transforms_list.append(Compose(self.transform_list))
            self.K = self.transforms["K"]
        
        if logger is not None:
            logger.info("===> load data from {}".format(self.data_path))
            logger.info("===> the shape of sequence data: {}.".format(self.sequence.shape))
            logger.info("===> the shape of label: {}.".format(self.label.shape))
            if self.phase == "train" and self.transforms["USE_TRANSFORMS"]:
                logger.info("===> using transforms: {}\n".format(self.transform_type))
    

    def __getitem__(self, index):
        data_transformed_list = []
        if self.phase == "train" and self.transforms["USE_TRANSFORMS"]:
            for transform in self.transforms_list:
                for _ in range(self.K):
                    data_transformed = transform(self.sequence[index])
                    data_transformed_list.append(data_transformed)
        else:
            for i in range(2):
                data_transformed_list.append(self.sequence[index])
        
        return self.sequence[index], self.label[index], data_transformed_list
    

    def __len__(self):
        return self.label.shape[0]

import random
import numpy as np
from torch.utils.data.sampler import Sampler

class RandomCycleIter:
    
    def __init__ (self, data, test_mode=False):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.i = self.length - 1
        self.test_mode = test_mode


    def __iter__ (self):
        return self
    

    def __next__ (self):
        self.i += 1
        
        if self.i == self.length:
            self.i = 0
            if not self.test_mode:
                random.shuffle(self.data_list)
            
        return self.data_list[self.i]


def class_aware_sample_generator (cls_iter, data_iter_list, n, num_samples_cls=1):
    i = 0
    j = 0

    while i < n:       
        if j >= num_samples_cls:
            j = 0
    
        if j == 0:
            temp_tuple = next(zip(*[data_iter_list[next(cls_iter)]]*num_samples_cls))
            yield temp_tuple[j]
        else:
            yield temp_tuple[j]
        
        i += 1
        j += 1


class ClassAwareSampler(Sampler):
    """class aware sampler

    """

    def __init__(self, dataset, num_samples_cls):
        num_classes = len(np.unique(dataset.label))  # num of classes: int
        
        self.class_iter = RandomCycleIter(range(num_classes))  # class iterator: select one class from all the classes

        cls_data_list = [list() for _ in range(num_classes)]  # data index of each class
        for i, label in enumerate(dataset.label):
            cls_data_list[label].append(i)

        self.data_iter_list = [RandomCycleIter(x) for x in cls_data_list]
        self.num_samples = max([len(x) for x in cls_data_list]) * len(cls_data_list)
        self.num_samples_cls = num_samples_cls

    
    def __iter__(self,):
        return class_aware_sample_generator(self.class_iter, self.data_iter_list,
                                            self.num_samples, self.num_samples_cls)

    
    def __len__ (self):
        return self.num_samples

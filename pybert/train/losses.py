#encoding:utf-8
import torch
from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss

from pybert.config.basic_config import configs as config

__call__ = ['CrossEntropy','BCEWithLogLoss']

class CrossEntropy(object):
    def __init__(self):
        self.loss_f = CrossEntropyLoss()

    def __call__(self, output, target):
        loss = self.loss_f(input=output, target=target)
        return loss

class BCEWithLogLoss(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_weights = torch.FloatTensor([0.75 , 0.6 , 0.51219512, 0.5060241 , 1. ,
                                            0.95454545, 0.31343284, 0.95454545, 0.71186441, 0.82352941]).to(self.device)
        # self.loss_fn = BCEWithLogitsLoss(weight=self.class_weights.repeat((config['train']['batch_size'],1)))
        self.loss_fn = BCEWithLogitsLoss()

    def __call__(self,output,target):
        loss = self.loss_fn(input = output,target = target)
        return loss

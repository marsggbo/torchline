#coding=utf-8
import torch
import torch.nn.functional as F
import numpy as np
from .build import LOSS_FN_REGISTRY

__all__ = [
    'FocalLoss'
]

@LOSS_FN_REGISTRY.register()
class FocalLoss(torch.nn.Module):    
    def __init__(self, cfg):
        """focal_loss function: -α(1-yi)**γ *ce_loss(xi,yi)   
        Args:
            alpha:  class weight (default 0.25).
                    When α is a 'list', it indicates the class-wise weights;
                    When α is a constant, 
                        if in detection task, it indicates that the class-wise weights are[α, 1-α, 1-α, ...], 
                                                the first class indicates the background
                        if in classification task, it indicates that the class-wise weights are the same
            gamma:  γ (default 2), focusing paramter smoothly adjusts the rate at which easy examples are down-weighted.
            num_classes: the number of classes
            size_average:  (default 'mean'/'sum') specify the way to compute the loss value
        """

        super(FocalLoss,self).__init__()
        self.cfg = cfg
        alpha = cfg.loss.focal_loss.alpha
        gamma = cfg.loss.focal_loss.gamma
        self.size_average = cfg.loss.focal_loss.size_average
        num_classes = cfg.model.classes
        if isinstance(alpha,list):
            assert len(alpha)==num_classes
            alpha /= np.sum(alpha) # setting the value in range of [0, 1]
            # print("Focal loss alpha = {}, assign specific weights for each class".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<=1   
            self.alpha = torch.zeros(num_classes)+0.00001

            # classification task
            # print("Focal loss alpha={}, the weight for each class is the same".format(alpha))
            self.alpha  += alpha

            # detection task # 如果α为一个常数,则降低第一类的影响,在目标检测中背景为第一类
            # print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            # self.alpha[0] += alpha
            # self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, predictions, labels):
        """
        focal_loss损失计算
        Args:        
            preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数        
            labels:  实际类别. size:[B,N] or [B]        
        return:
            loss
        """        
        assert predictions.dim()==2 and labels.dim()==1        
        preds = predictions.view(-1,predictions.size(-1))  # num*classes
        alpha = self.alpha.to(labels.device) 

        # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)   
        preds_softmax = F.softmax(preds, dim=1)      
        preds_logsoft = torch.log(preds_softmax)

        # implement nll_loss ( crossempty = log_softmax + nll )     
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))  # num*1    
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))  # num*1      
        alpha = alpha.gather(0,labels.view(-1)) # num     

        # calc loss
        # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        # shape: num*1
        loss = -torch.mul( torch.pow((1-preds_softmax), self.gamma), preds_logsoft )  
        # α * (1-pt)**γ * ce_loss
        # shape: 
        loss = torch.mul(alpha, loss.t())    
        del preds    
        del alpha
        if self.size_average:        
            loss = loss.mean()        
        else:            
            loss = loss.sum()        
        return loss
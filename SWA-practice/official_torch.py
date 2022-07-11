import itertools
import math
from copy import deepcopy
'''
[reference] https://github.com/pytorch/pytorch/blob/master/torch/optim/swa_utils.py
'''

import warnings

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler

# _LRScheduler: torch의 LambdaLR, MultiplicativeLR 등 여러 learning rate 관련 클래스들이 상속받는 기본 클래스

__all__ = ['AveragedModel', 'update_bn', 'SWALR']

class AveragedModel(nn.Module):
    '''
    [Averaging Weights Leads to Wider Optima and Better Generalzation UAI. 2018] 에서 소개된
    Stochastic Weight Averaging를 구현한 것

    AveragedModel 클래스를 통해서 주어진 'model'에 대한 copy를 'device'에 만들어주고,
    'model'의 parameter의 평균을 계산할 수 있게 해줌

    Args:
        -- model        (torch.nn.Module)       : SWA를 적용할 모델
        -- device       (torch.device, optional): 'cuda'나 'cpu'에 모델이 올라가게 됨
        -- avg_fn       (function, optional)    : 파라미터를 update할 때 사용하는 averaging 함수
                                                : 반드시 AveragedModel, model,의 current parameter가 들어가야 됨
        -- use_buffers  (bool, default; False)  : True면 model 파라미터 뿐 아니라 buffer에 대한 평균도 계산함

    note ::
        Batch Normalization이 들어가 있는 모델에 SWA를 사용할 경우에는 activation statistics를 update 해주어야함
        torch.optim.swa_utils.update_bn 을 사용하거나 'use_buffers'를 True로 설정하면 됨.

        1) 첫번째는 post-training 방식으로 model에 data를 보내주면서 statistics update
        2) parameter update phase에서 모든 buffer를 평균해주면서 수행함

        경험적으로는 statistics update 해주는 것이 성능을 증가시켰지만, 본인 case에 맞게 update 하는 것이 좋은 지 아닌 지 파악필요.
    '''
    def __init__(self, model, device=None, avg_fn=None, use_buffers=None):
        super(AveragedModel, self).__init__()

        # deep copy를 통해서 내부의 객체들까지 새롭게 copy 됨
        self.module = deepcopy(model) 

        # device가 주어져있다면 model을 'cuda'나 'cpu'로 보냄 ('cuda' 겠지만 대부분)
        if device is not None:
            self.module = self.module.to(device)

        # 일반적으로 모델의 parameter로 사용되지 않는 buffer를 등록할 때 사용 (ex. BatchNorm에서 running_mean)
        self.register_buffer('n_averaged', torch.tensor(0, dtype=torch.long, device=device))

        # 논문에 있는 기본식, customize 해서 사용할 수도 있음
        if avg_fn is None:
            def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
                return averaged_model_parameter + \
                    (model_parameter - averaged_model_parameter) / (num_averaged + 1)

        self.avg_fn = avg_fn
        self.use_buffers = use_buffers

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    # 해당 함수를 통해서 p_swa를 p_swa, p_model을 통해 update해주는 과정
    def update_parameters(self, model):
        self_param = (
            itertools.chain(self.module.parameters(), self.module.buffers()) # itertools.chain : 여러 list를 알아서 unpack하고 붙혀줌
            if self.use_buffers else self.parameters()
        )
        model_param = (
            itertools.chain(model.parameters(), model.buffers())
            if self.use_buffers else model.parameters()
        )
        for p_swa, p_model in zip(self_param, model_param):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(self.avg_fn(p_swa.detach(), p_model_,
                self.n_averaged.to(device)))

        self.n_averaged += 1

@torch.no_grad()
def update_bn(loader, model, device=None):
    '''
    BatchNorm의 running_mean, running_var buffers를 update 해주는 함수

    'loader'에 들어있는 data를 모델에 한번 pass시켜서 statistics를 estimate 함

    Args:
        -- loader   (torch.utils.data.DataLoader)   : activation statistics를 계산하기 위해 넣어주는 data loader, tensor 형태여야 함
        -- model    (torch.nn.Module)               : 들어갈 model
        -- device   (torch.device, optional)        : 'cuda'
    '''
    momenta = {}

    # module 내의 BatchNorm layer를 초기화 해주는 부분
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training # model이 evaluation mode라면 False를 반환함. 일종의 flag로 사용
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for input in loader:
        if isinstance(input, (list, tuple)):
            input = input[0]
        if device is not None:
            input = input.to(device)

        model(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]

    model.train(was_training)


class SWALR(_LRScheduler):
    '''
    각 파라미터의 learning rate를 fixed value로 조정(annealing)해줌

    Args:
        -- optimizer            (torch.optim.Optimizer) : wrapped optimizer
        -- swa_lrs              (float or list)         : 모든 parameter group에 동일하게 적용될 learning rate 또는 group 별로 다른 learning rate
        -- annealing_epohcs     (int, default; 10)      : annealing phase의 epoch 수
        -- annealing_strategy   (str, default; 'cos')   : | 'cos' | 'linear' |
        -- lalst_epoch          (int, default; -1)      : 마지막 epoch의 index

    note ::
        학습 초반에는 다른 scheduler로 학습하다가 중반부터 SWA로 학습하는 것도 가능

        ex) 
        if i > swa_start:
            swa_scheduler.step()
        else:
            scheduler.step()
    '''
    def __init__(self, optimizer, swa_lr, anneal_epochs=10, anneal_strategy='cos', last_epoch=-1):
        swa_lrs = self._format_param(optimizer, swa_lr)
        for swa_lr, group in zip(swa_lrs, optimizer.param_groups):
            group['swa_lr'] = swa_lr
        if anneal_strategy not in ['cos', 'linear']:
            raise ValueError('anneal_strategy must be one of "cos" or "linear"' f'instead got {anneal_strategy}')
        
        elif anneal_strategy =='cos':
            self.anneal_func = self._cosine_anneal
        elif anneal_strategy =='linear':
            self.anneal_func = self._linear_anneal
        
        if not isinstance(anneal_epochs, int) or anneal_epochs < 0:
            raise ValueError(f'anneal_epochs must be equal or greater than 0, got {anneal_epochs}')
        self.anneal_epochs = anneal_epochs
        super(SWALR, self).__init__(optimizer, last_epoch)

    @staticmethod
    def _format_param(optimizer, swa_lrs):
        if isinstance(swa_lrs, (list, tuple)):
            if len(swa_lrs) != len(optimizer.param_groups):
                raise ValueError('swa_lr must have the same length as '
                f'optimizer.param_groups : swa_lr has {len(swa_lrs)},'
                f'optimizer.param_groups has {len(optimizer.param_groups)}')
            
            return swa_lrs
        else:
            return [swa_lrs] * len(optimizer.param_groups)

    
    @staticmethod
    def _linear_anneal(t):
        return t

    @staticmethod
    def _cosine_anneal(t):
        return (1 - math.cos(math.pi * t)) / 2

    @staticmethod
    def _get_initial_lr(lr, swa_lr, alpha):
        if alpha == 1:
            return swa_lr
        return (lr - alpha * swa_lr) / (1 - alpha)

    
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        step = self._step_count - 1
        if self.anneal_epochs == 0:
            step = max(1, step)
        prev_t = max(0, min(1, (step - 1) / max(1, self.anneal_epochs)))
        prev_alpha = self.anneal_func(prev_t)
        prev_lrs = [self._get_initial_lr(group['lr'], group['swa_lr'], prev_alpha)
                    for group in self.optimizer.param_groups]
        t = max(0, min(1, step / max(1, self.anneal_epochs)))
        alpha = self.anneal_func(t)
        return [group['swa_lr'] * alpha + lr * (1 - alpha)
                for group, lr in zip(self.optimizer.param_groups, prev_lrs)]
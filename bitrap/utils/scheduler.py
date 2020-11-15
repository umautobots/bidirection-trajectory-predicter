'''
some schedulers used for scheduling hyperparameters over training procedure
Adopted from Trajectron++
'''

import torch
import torch.optim as optim
import functools

import warnings
import pdb

class CustomLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        super(CustomLR, self).__init__(optimizer, lr_lambda, last_epoch)

    def get_lr(self):
        return [lmbda(self.last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]

class ParamScheduler():
    def __init__(self):
        self.schedulers = []
        self.annealed_vars = []

    def create_new_scheduler(self, name, annealer, annealer_kws, creation_condition=True):
        value_scheduler = None
        rsetattr(self, name + '_scheduler', value_scheduler)
        if creation_condition:
            value_annealer = annealer(annealer_kws)
            rsetattr(self, name + '_annealer', value_annealer)

            # This is the value that we'll update on each call of
            # step_annealers().
            rsetattr(self, name, value_annealer(0).clone().detach())
            dummy_optimizer = optim.Optimizer([rgetattr(self, name)], {'lr': value_annealer(0).clone().detach()})
            rsetattr(self, name + '_optimizer', dummy_optimizer)
            value_scheduler = CustomLR(dummy_optimizer,
                                        value_annealer)
            rsetattr(self, name + '_scheduler', value_scheduler)

        self.schedulers.append(value_scheduler)
        self.annealed_vars.append(name)

    def step(self):
        # This should manage all of the step-wise changed
        # parameters automatically.
        for idx, annealed_var in enumerate(self.annealed_vars):
            if rgetattr(self, annealed_var + '_scheduler') is not None:
                # First we step the scheduler.
                with warnings.catch_warnings():  # We use a dummy optimizer: Warning because no .step() was called on it
                    warnings.simplefilter("ignore")
                    rgetattr(self, annealed_var + '_scheduler').step()

                # Then we set the annealed vars' value.
                rsetattr(self, annealed_var, rgetattr(self, annealed_var + '_optimizer').param_groups[0]['lr'])

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def sigmoid_anneal(anneal_kws):
    device = anneal_kws['device']
    start = torch.tensor(anneal_kws['start'], device=device)
    finish = torch.tensor(anneal_kws['finish'], device=device)
    center_step = torch.tensor(anneal_kws['center_step'], device=device, dtype=torch.float)
    steps_lo_to_hi = torch.tensor(anneal_kws['steps_lo_to_hi'], device=device, dtype=torch.float)
    return lambda step: start + (finish - start)*torch.sigmoid((torch.tensor(float(step), device=device) - center_step) * (1./steps_lo_to_hi))

from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.optim import Adam, SGD
import tqdm
import torch
import json


def get_num_layer_for_vit(var_name, num_max_layer):
    if 'embedding' in var_name or 'conv1' in var_name or 'ln_pre' in var_name:
        return 0
    elif 'resblocks' in var_name:
        layer_id = int(var_name.split('.')[5])
        return layer_id + 1
    elif "classifier" in var_name:
        return num_max_layer - 1
    else:
        return num_max_layer - 2


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))

def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay
        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, group_name)
        else:
            layer_id = None

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:   
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            if scale > 0:
                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr": scale,
                    # "lr_scale": scale,
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    # "lr_scale": scale,
                    "lr": scale,
                    "name": group_name
                }
        if scale > 0:
            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)
    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())



def epoch_wrapup(pl_module: LightningModule, mode: str):
    r"""On the end of each epoch, log information of the whole
        epoch and reset all metrics.
    
    Args:
        pl_module: An instance of LightningModule.
        mode: The current mode (train, val or test).
    """
    assert mode in ["train", "val", "test"]
    value = getattr(pl_module, f"{mode}_loss").compute()
    if mode == 'train':
        pl_module.log(f"{mode}/loss_epoch", value)
    getattr(pl_module, f"{mode}_loss").reset()
    value = getattr(pl_module, f"{mode}_acc").compute()
    if mode == 'train':
        pl_module.log(f"{mode}/acc_epoch", value)
    getattr(pl_module, f"{mode}_acc").reset()

def set_schedule(pl_module):
    r"""Set the optimizer and scheduler for training.

    Supported optimizer:
        Adam and SGD
    Supported scheduler:
        cosine scheduler and decaying on specified epochs

    Args:
        pl_module: An instance of LightningModule.
    """
    lr = pl_module.hparams.lr
    wd = pl_module.hparams.weight_decay
    decay_scheduler = pl_module.hparams.decay_scheduler
    optim_type = pl_module.hparams.optim_type


    if optim_type == "adam":
        optimizer = Adam(pl_module.parameters(),
                                    weight_decay=wd, lr=lr)
    elif optim_type == "sgd":
        optimizer = SGD(pl_module.parameters(),
                                    momentum=0.9, nesterov=True,
                                    weight_decay=wd, lr=lr)
    elif "layerwise" in optim_type:
        layer_decay = 0.65
        num_layers = pl_module.backbone.model.visual.transformer.layers
        # assigner = LayerDecayValueAssigner(list((layer_decay ** (num_layers + 1 - i)) for i in range(num_layers + 2)))
        # assigner = LayerDecayValueAssigner(list(0.00005 for i in range(num_layers + 2)))
        # assigner = LayerDecayValueAssigner(list(lr * (layer_decay ** (num_layers + 1 - i)) for i in range(num_layers + 2)))
        # assigner = LayerDecayValueAssigner([5e-9, 5e-9, 5e-9, 5e-9, 5e-9, 5e-9, 5e-9, 5e-9, 5e-9, 5e-9, 5e-9, 5e-9, 5e-5, 0.07])
        # assigner = LayerDecayValueAssigner([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1e-4, 0.5, 1])
        assigner = LayerDecayValueAssigner(lr)
        # assigner = LayerDecayValueAssigner([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, lr])
        get_num_layer = assigner.get_layer_id
        get_layer_scale = assigner.get_scale
        parameters = get_parameter_groups(pl_module, weight_decay=wd, get_num_layer=get_num_layer, get_layer_scale=get_layer_scale)
        if "sgd" in optim_type:
            optimizer = SGD(parameters, momentum=0.9, nesterov=True, weight_decay=wd, lr=5e-5)
        elif "adam" in optim_type:
            optimizer = Adam(parameters, weight_decay=wd, lr=5e-5)
    else:
        raise RuntimeError("optim_type not supported.\
                            Try to implement your own optimizer.")
    
    if decay_scheduler == "cosine":
        if pl_module.trainer.max_steps is None:
            length_epoch = len(pl_module.trainer.datamodule.train_dataloader())
            max_steps = length_epoch * pl_module.trainer.max_epochs
            print(f"length_epoch:{length_epoch}")
            print(f"max_epochs:{pl_module.trainer.max_epochs}")
        else:
            max_steps = pl_module.trainer.max_steps
        
        scheduler = {'scheduler': CosineAnnealingLR(optimizer,max_steps),
                     'interval': 'step'}
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    elif decay_scheduler == "specified_epochs":
        decay_epochs = pl_module.hparams.decay_epochs
        decay_power = pl_module.hparams.decay_power
        assert decay_epochs is not None and decay_power is not None
        scheduler = {'scheduler': 
                     MultiStepLR(optimizer, milestones=decay_epochs, gamma=decay_power),
                     'interval': 'epoch'}
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    elif decay_scheduler is None:
        return optimizer
    else:
        raise RuntimeError("decay scheduler not supported.\
                            Try to implement your own scheduler.")


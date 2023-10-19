import sys
from sacred import Experiment
import yaml
import time 
import os
import lr.lr_config as lr_config
import json
method, task, taskid, wandb_project, module, logdir, cscale, cnumber, save_dir, train_val_dataset, train_val_data_path, test_dataset, test_data_path, learn_rate, backbone, epoch, optim, load_head, shot, dataseed = sys.argv[1:]
ex = Experiment("ProtoNet", save_git_info=False)

# train_config = getattr(lr_config, learn_rate)
train_config = getattr(lr_config, f"{train_val_dataset[5:]}_{backbone}_config")

full_config = getattr(lr_config, learn_rate)

def convert_to_e(lr):
    res = "{:.0e}".format(lr)
    if lr >= 1:
        res = res.split("+")
    else:
        res = res.split("-")
    if lr < 10:
        return f"head_{res[0]}_{str(int(res[1]))}"
    else:
        return f"head_{res[0]}{str(int(res[1]))}"

@ex.config
def config():
    config_dict = {}

    #if training CLIP, set to True
    config_dict["load_pretrained"] = True
    #if training, set to False
    config_dict["is_test"] = False
    if config_dict["is_test"]:
        #if testing, specify the total rounds of testing. Default: 5
        config_dict["num_test"] = 5
        config_dict["load_pretrained"] = True
        #specify pretrained path for testing.
    if config_dict["load_pretrained"]:
        print("---------------------------load head------------------------------------")
        print(load_head)
        print("---------------------------load head------------------------------------")
        if load_head == "False":
            print("load head=False")
            config_dict["pre_trained_path"] = None
        elif load_head == "True":
            # load the pretrained head
            config_dict["only_load_classifier"] = True
            try:
                if isinstance(train_config["head_lr"], dict):
                    head_lr = float(train_config["head_lr"][shot])
                else:
                    head_lr = float(train_config["head_lr"])
            except:
                head_lr = float(train_config["lr"])
            head_lr = convert_to_e(head_lr)
            # pretrain_dir = os.path.join("head_result", f"{train_val_dataset[5:]}", f"CoOp_layerwise_partialft_{train_val_dataset[5:]}_{backbone}_clip_FT_CoOp_head_lr_{head_lr}", "CoOp", "CoOp_layerwise_partialft", f"{train_val_dataset[5:]}_epoch={epoch}_seed_{dataseed}_shot_{shot}_{cscale}_clip_FT_CoOp_head_{cnumber}_lr_{head_lr}", "checkpoints")
            # pretrain_dir = os.path.join(f"ceph_result/head_{backbone}_clip_FT_CoOp_head", f"{train_val_dataset[5:]}", f"CoOp_layerwise_partialft_{train_val_dataset[5:]}_{backbone}_clip_FT_CoOp_head_lr_{train_val_dataset[5:]}_config", "CoOp", "CoOp_layerwise_partialft", f"{train_val_dataset[5:]}_epoch={epoch}_seed_{dataseed}_shot_{shot}_{cscale}_clip_FT_CoOp_head_{cnumber}_lr_{train_val_dataset[5:]}_config", "checkpoints")
            pretrain_dir = os.path.join(f"ceph_result/head_{backbone}_clip_FT_CoOp_head", f"{train_val_dataset[5:]}", f"CoOp_layerwise_partialft_{train_val_dataset[5:]}_{backbone}_clip_FT_CoOp_head_lr_{head_lr}", "CoOp", "CoOp_layerwise_partialft", f"{train_val_dataset[5:]}_epoch={epoch}_seed_{dataseed}_shot_{shot}_{cscale}_clip_FT_CoOp_head_{cnumber}_lr_{head_lr}", "checkpoints")
            allckpt = os.listdir(pretrain_dir)
            for ckpt in allckpt:
                if "epoch" in ckpt:
                    bestckpt = ckpt
            config_dict["pre_trained_path"] = os.path.join(pretrain_dir, bestckpt)
        else:
            raise ValueError("load head can only be True or False")
        #only load the backbone.
        config_dict["load_backbone_only"] = False
        
    #Specify the model name, which should match the name of file
    #that contains the LightningModule
    config_dict["model_name"] = module
 
    

    #whether to use multiple GPUs
    multi_gpu = False
    if config_dict["is_test"]:
        multi_gpu = False
    #The seed
    seed = 10
    config_dict["seed"] = seed

    #The logging dirname: logdir/exp_name/
    log_dir = logdir
    exp_name = f"{method}/{task}"
    # exp_name = os.path.join(branch, exp_name)
    
    #Three components of a Lightning Running System
    trainer = {}
    data = {}
    model = {}


    ################trainer configuration###########################


    ###important###

    #debugging mode
    trainer["fast_dev_run"] = False

    if multi_gpu:
        trainer["accelerator"] = "ddp"
        trainer["sync_batchnorm"] = True
        trainer["gpus"] = [2,3]
        trainer["plugins"] = [{"class_path": "plugins.modified_DDPPlugin"}]
    else:
        trainer["accelerator"] = None
        trainer["gpus"] = [0]
        trainer["sync_batchnorm"] = False
    
    # whether resume from a given checkpoint file
    trainer["resume_from_checkpoint"] = None # example: "../results/ProtoNet/version_11/checkpoints/epoch=2-step=1499.ckpt"

    # The maximum epochs to run
    trainer["max_epochs"] = epoch

    # potential functionalities added to the trainer.
    trainer["callbacks"] = [{"class_path": "pytorch_lightning.callbacks.LearningRateMonitor", 
                  "init_args": {"logging_interval": "step"}
                  },
                {"class_path": "pytorch_lightning.callbacks.ModelCheckpoint",
                  "init_args":{"verbose": True, "save_last": True, "monitor": "val/acc", "mode": "max"}
                },
                {"class_path": "callbacks.SetSeedCallback",
                 "init_args":{"seed": seed, "is_DDP": multi_gpu}
                }]

    ###less important###
    num_gpus = trainer["gpus"] if isinstance(trainer["gpus"], int) else len(trainer["gpus"])
    trainer["logger"] = {
        "class_path":"pytorch_lightning.loggers.WandbLogger",
        "init_args": {
            "save_dir": os.path.join(log_dir, exp_name), 
            "log_model": True, 
            "project": wandb_project,
            "name": f"{method}/{task}/{taskid}", 
            "id": taskid
        }
    }
    trainer["log_every_n_steps"]=1
    trainer["replace_sampler_ddp"] = False

    

    ##################shared model and datamodule configuration###########################

    #important
    per_gpu_train_batchsize = 256
    if backbone == "ViT_B_16_clip":
        per_gpu_train_batchsize = 64
    # train_shot = 5
    # test_shot = 1

    #less important
    per_gpu_val_batchsize = 512
    per_gpu_test_batchsize = 512
    # train_way = 5
    # val_way = 5
    # test_way = 5
    # val_shot = 1
    # num_query = 15

    ##################datamodule configuration###########################

    #important

    #The name of dataset, which should match the name of file
    #that contains the datamodule.
    data["is_CoOp"] = True
    data["backbone_name"]=backbone
    data["train_dataset_name"] = train_val_dataset

    data["train_data_root"] = train_val_data_path

    data["val_test_dataset_name"] = train_val_dataset

    data["val_test_data_root"] = train_val_data_path

    data["test_dataset_name"] = test_dataset

    data["test_data_root"] = test_data_path
    #determine whether meta-learning.
    data["is_meta"] = True
    if data["is_CoOp"]:
        data["is_meta"] = False
    data["train_num_workers"] = 12
    #the number of tasks
    # data["train_num_task_per_epoch"] = 20
    # data["val_num_task"] = 1000
    # data["test_num_task"] = 2000
    
    alltrain_num = len(json.load(open(data["train_data_root"], "r"))["train"]["data"])
    
    #less important
    if alltrain_num < num_gpus*per_gpu_train_batchsize:
        data["train_batchsize"] = alltrain_num
    else:
        data["train_batchsize"] = num_gpus*per_gpu_train_batchsize
    data["val_batchsize"] = num_gpus*per_gpu_val_batchsize
    data["test_batchsize"] = num_gpus*per_gpu_test_batchsize
    # data["test_shot"] = test_shot
    # data["train_shot"] = train_shot
    data["val_num_workers"] = 16
    data["is_DDP"] = True if multi_gpu else False
    # data["train_way"] = train_way
    # data["val_way"] = val_way
    # data["test_way"] = test_way
    # data["val_shot"] = val_shot
    # data["num_query"] = num_query
    data["drop_last"] = False

    ##################model configuration###########################

    #important

    #The name of feature extractor, which should match the name of file
    #that contains the model.
    model["backbone_name"] = backbone
    #the initial learning rate
    # model["lr"] = float(learn_rate)*data["train_batchsize"]/256
    # model["lr"] = euro_vit_B_32["lr"]*(data["train_batchsize"]/256)
    lr_scale = data["train_batchsize"]/256
    # model["lr"] = [(lr_scale * i) for i in euro_vit_B_32["lr"]]

    try:
        if "head" in config_dict["model_name"]:
            if isinstance(train_config["head_lr"], dict):
                head_lr = float(train_config["head_lr"][shot])
            else:
                head_lr = float(train_config["head_lr"])
            model["lr"] = head_lr*lr_scale
        else:
            if full_config != None:
                model['lr'] = full_config['lr']*lr_scale
            else:
                if isinstance(train_config["full_lr"], float):
                    model["lr"] = train_config["full_lr"]*lr_scale
                elif isinstance(train_config["full_lr"], list):
                    model["lr"] = []
                    for i in range(len(train_config["full_lr"])):
                        model["lr"].append(train_config["full_lr"][i]*lr_scale)
                elif isinstance(train_config["full_lr"], dict):
                    model["lr"] = train_config["full_lr"][shot]*lr_scale
                else:
                    raise ValueError("lr should be either float or list.")
    except:
        model["lr"] = train_config["lr"]*lr_scale

    #less important
    # model["train_way"] = train_way
    # model["val_way"] = val_way
    # model["test_way"] = test_way
    # model["train_shot"] = train_shot
    # model["val_shot"] = val_shot
    # model["test_shot"] = test_shot
    # model["num_query"] = num_query
    model["train_batch_size_per_gpu"] = per_gpu_train_batchsize
    model["val_batch_size_per_gpu"] = per_gpu_val_batchsize
    model["test_batch_size_per_gpu"] = per_gpu_test_batchsize
    model["weight_decay"] = 5e-4
    #The name of optimization scheduler
    model["decay_scheduler"] = "cosine"
    # if model["decay_scheduler"] is "specified_epochs":
    #     model["decay_epochs"] = [200,300,400,500]
    #     model["decay_power"] = 0.5
    model["optim_type"] = optim
    #cosine or euclidean
    model["metric"] = "cosine"
    model["scale_cls"] = 10.
    model["normalize"] = True
    model["cscale"] = float(cscale)
    model['cnumber'] = int(cnumber)
    model['cname'] = train_val_dataset[5:]

    config_dict["trainer"] = trainer
    config_dict["data"] = data
    config_dict["model"] = model



# @ex.automain
def main(_config):
    config_dict = _config["config_dict"]
    # file_ = 'config/config.yaml'
    file_ = os.path.join(save_dir, "config.yaml") 
    print(file_)
    stream = open(file_, 'w')
    yaml.safe_dump(config_dict, stream=stream,default_flow_style=False)

if __name__ == "__main__":
    _config = config()
    main(_config)
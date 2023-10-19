import sys
from sacred import Experiment
import yaml
import os
import lr.lr_config as lr_config
import json
method, task, taskid, wandb_project, ckptPath, module, cscale, cnumber, save_dir, train_val_dataset, train_val_data_path, test_dataset, test_data_path, learn_rate, backbone, logdir, shot = sys.argv[1:]
ex = Experiment("ProtoNet", save_git_info=False)

train_config = getattr(lr_config, learn_rate)

imageNet_var = ["imagenet-adversarial", "imagenet-rendition", "imagenet-sketch", "imagenetv2"]

@ex.config
def config():
    config_dict = {}

    #if training CLIP, set to True
    config_dict["load_pretrained"] = True
    #if training, set to False
    config_dict["is_test"] = True
    if config_dict["is_test"]:
        #if testing, specify the total rounds of testing. Default: 5
        config_dict["num_test"] = 1
        config_dict["load_pretrained"] = True
        #specify pretrained path for testing.
    if config_dict["load_pretrained"]:
        config_dict["pre_trained_path"] = ckptPath
        #only load the backbone.
        if test_dataset[5:] in imageNet_var:
            config_dict["load_backbone_only"] = False
        else:
            if test_dataset == train_val_dataset:
                config_dict["load_backbone_only"] = False
            else:
                config_dict["load_backbone_only"] = True
        
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
    exp_name = os.path.join(exp_name, wandb_project, taskid, test_dataset)
    
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
    trainer["max_epochs"] = 10

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
    trainer["logger"] = {"class_path":"pytorch_lightning.loggers.TensorBoardLogger",
                        "init_args": {"save_dir": log_dir,"name": exp_name, "version": f"results"}
                        }
    trainer["replace_sampler_ddp"] = False

    

    ##################shared model and datamodule configuration###########################

    #important
    per_gpu_train_batchsize = 256
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
    
    if test_dataset in imageNet_var:
        data["test_dataset_name"] = "CoOp_imagevar"
    else:
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

    lr_scale = data["train_batchsize"]/256
    # model["lr"] = [(lr_scale * i) for i in euro_vit_B_32["lr"]]
    try:
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
    model["optim_type"] = "sgd"
    #cosine or euclidean
    model["metric"] = "cosine"
    model["scale_cls"] = 10.
    model["normalize"] = True
    model["cscale"] = float(cscale)
    model['cnumber'] = int(cnumber)
    model['cname'] = test_dataset[5:]
    
    config_dict["trainer"] = trainer
    config_dict["data"] = data
    config_dict["model"] = model



# @ex.automain
def main(_config):
    config_dict = _config["config_dict"]
    file_ = os.path.join(save_dir, test_dataset, "config.yaml") 
    stream = open(file_, 'w')
    yaml.safe_dump(config_dict, stream=stream,default_flow_style=False)

if __name__ == "__main__":
    _config = config()
    main(_config)
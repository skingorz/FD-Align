# generate the context using zeroclip and generate the context using kmeans
from torch.nn import Module
import torch
from architectures import get_backbone, clip_head, CoOp_head
import torch.nn.functional as F
from . import utils
from .base_module import BaseFewShotModule
from typing import Tuple, List, Optional, Union, Dict
from architectures.feature_extractor.clip import ImageEncoder, load, get_ILF_kmeans_weights_classifier, get_zeroshot_weight
from dataset_and_process.datasets.openai_imagenet_temple import openai_imagenet_template
# from dataset_and_process.datasets.class_name import openai_imageNet_classnames, mini_val
import dataset_and_process.datasets.class_name as class_name
class CLIP_context(BaseFewShotModule):
    r"""The datamodule implementing Prototypical Network.
    """
    def __init__(
        self,
        metric: str = "cosine",
        scale_cls: float = 10.,
        normalize: bool = True,
        backbone_name: str = "resnet12",
        train_way: int = 5,
        val_way: int = 5,
        test_way: int = 5,
        train_shot: int = 5,
        val_shot: int = 5,
        test_shot: int = 5,
        num_query: int = 15,
        train_batch_size_per_gpu: int = 2,
        val_batch_size_per_gpu: int = 2,
        test_batch_size_per_gpu: int = 2,
        lr: float = 0.1,
        weight_decay: float = 5e-4,
        decay_scheduler: Optional[str] = "cosine",
        optim_type: str = "sgd",
        decay_epochs: Union[List, Tuple, None] = None,
        decay_power: Optional[float] = None,
        local_rank: int = -1,
        backbone_kwargs: Dict = {},
        cscale: float = 1.0,
        cnumber: int = 1,
        cname: str = "openai_imageNet_classnames",
        **kwargs
    ) -> None:
        """   
        Args:
            metric: what metrics applied. "cosine" or "euclidean".
            scale_cls: The initial scale number which affects the 
                    following softmax function.
            normalize: Whether normalize each spatial dimension of image features before average pooling.
            backbone_name: The name of the feature extractor, 
                        which should match the correspond 
                        file name in architectures.feature_extractor
            train_way: The number of classes within one training task.
            val_way: The number of classes within one training task.
            test_way: The number of classes within one training task.
            train_shot: The number of samples within each few-shot 
                        support class during training. 
                        For meta-learning only.
            val_shot: The number of samples within each few-shot 
                    support class during validation.
            test_shot: The number of samples within each few-shot 
                    support class during testing.
            num_query: The number of samples within each few-shot 
                    query class.
            train_batch_size_per_gpu: The batch size of training per GPU.
            val_batch_size_per_gpu: The batch size of validation per GPU.
            test_batch_size_per_gpu: The batch size of testing per GPU.
            lr: The initial learning rate.
            weight_decay: The weight decay parameter.
            decay_scheduler: The scheduler of optimizer.
                            "cosine" or "specified_epochs".
            optim_type: The optimizer type.
                        "sgd" or "adam"
            decay_epochs: The list of decay epochs of decay_scheduler "specified_epochs".
            decay_power: The decay power of decay_scheduler "specified_epochs"
                        at eachspeicified epoch.
                        i.e., adjusted_lr = lr * decay_power
            backbone_kwargs: The parameters for creating backbone network.
        """
        super().__init__(
            backbone_name, train_way, val_way, test_way, train_shot, val_shot,
            test_shot, num_query, train_batch_size_per_gpu,
            val_batch_size_per_gpu, test_batch_size_per_gpu,
            lr, weight_decay, decay_scheduler, optim_type,
            decay_epochs, decay_power, backbone_kwargs
        )
        # self.classifier = clip_head()
        
        clip_model = ImageEncoder(backbone_name)
        clip_model_, _, _ = load(backbone_name, jit=False)
        classname = getattr(class_name, cname)
        zero_shot_weight = get_zeroshot_weight(clip_model_, openai_imagenet_template, classname)
        self.classifier = CoOp_head(normalize=True, weights=zero_shot_weight)
        self.scale = cscale
        # self.context_scale = torch.nn.Parameter(torch.FloatTensor(1).fill_(1), requires_grad=True)


        self.zero_shot_clip = clip_model
        for param in self.zero_shot_clip.parameters():
            param.requires_grad = False
        self.context_classifier = get_ILF_kmeans_weights_classifier(clip_model_, openai_imagenet_template, classname, cnumber, 60)
        for param in self.classifier.parameters():
            param.requires_grad = True

        for param in self.context_classifier.parameters():
            param.requires_grad = False

        del clip_model_
        self.loss_ctx = torch.nn.KLDivLoss()


    def forward(self, batch):
        r"""fine tune based on cross entropy loss.
        
        Args:
            batch: a batch from val_dataloader.
        """
        # num_support_samples = way * shot
        data, lb = batch
        data = self.backbone(data)
        logits = self.classifier(data)

        # if len(data.shape) == 2:
        #     data = data.reshape([batch_size, -1] + list(data.shape[-1:]))
        # else:
        #     data = data.reshape([batch_size, -1] + list(data.shape[-3:]))
        # data_support = data[:, :num_support_samples]
        # data_query = data[:, num_support_samples:]
        # logits = self.classifier(data_query, data_support, way, shot)
        return logits, lb

    def train_forward(self, batch):
        image, lb = batch
        data = self.backbone(image)
        with torch.no_grad():
            zero_data = self.zero_shot_clip(image)

        # context KL loss
        # use KL loss compute the context loss between the data and the zero shot data

        # add normalization is unnecessary
        # data = F.normalize(data, dim=1)
        # zero_data = F.normalize(zero_data, dim=1)
        ctx_loss = self.loss_ctx(torch.log(F.softmax(self.context_classifier(data), dim=1)), F.softmax(self.context_classifier(zero_data), dim=1))

        ctx_loss = self.scale * ctx_loss

        # classification logits
        logits = self.classifier(data)


        return logits, lb, ctx_loss
        
    def val_test_forward(self, batch):
        return self(batch)
    
    def shared_step(self, batch, mode):
        r"""The shared operation across
            validation, testing and potentially training (meta-learning).
            
        Args:
            batch: a batch from val_dataloader.
            mode: train, val or test
        """
        assert mode in ["train", "val", "test"]
        if mode == "train":
            flag = "train"
        else:
            flag = "val_test"
        foward_function = getattr(self, f"{flag}_forward")
        batch_size_per_gpu = getattr(self.hparams, f"{mode}_batch_size_per_gpu")
        shot = getattr(self.hparams, f"{mode}_shot")

        way = getattr(self.hparams, f"{mode}_way")
        if mode == "train":
            logits, label, ctx_loss = foward_function(batch)
            self.log(f"{mode}/ctx_loss", ctx_loss)
        else:
            logits, label = foward_function(batch)
        # label = getattr(self, f"{mode}_label")
        # label = torch.unsqueeze(label, 0).repeat(batch_size_per_gpu, 1).reshape(-1).to(logits.device)
        # logits = logits.reshape(label.size(0),-1)
        
        loss = F.cross_entropy(logits, label)
        log_loss = getattr(self, f"{mode}_loss")(loss)
        accuracy = getattr(self, f"{mode}_acc")(logits, label)
        self.log(f"{mode}/loss", log_loss)
        self.log(f"{mode}/acc", accuracy)
        # self.log(f"{mode}/context_scale", (torch.exp(self.context_scale)-self.context_scale))
        if mode == "train":
            final_loss = loss + ctx_loss
            self.log(f"{mode}/all_loss", final_loss)
            return final_loss
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")
    
    def configure_optimizers(self):
        return utils.set_schedule(self)

def get_model():
    return CLIP_context
# Code ported from https://github.com/openai/CLIP

import hashlib
import os
import urllib
import warnings
from typing import Union, List
import pickle

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop
from tqdm import tqdm

from architectures.feature_extractor.model import build_model
from architectures.feature_extractor.tokenizer import SimpleTokenizer as _Tokenizer

__all__ = ["available_models", "load", "tokenize"]
_tokenizer = _Tokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT_B_32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT_B_16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}

def torch_load(save_path, device=None):
    with open(save_path, 'rb') as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


class ImageEncoder(torch.nn.Module):
    def __init__(self, model, device="cuda", keep_lang=False, jit=False):
        super().__init__()

        self.model, self.train_preprocess, self.val_preprocess = load(
            model, device=device, jit=False)
        
        # self.cache_dir = args.cache_dir

        if not keep_lang and hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    @classmethod
    def load(cls, filename):
        print(f'Loading image encoder from {filename}')
        return torch_load(filename)


def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target

def _convert_to_rgb(image):
    return image.convert('RGB')

def _transform(n_px: int, is_train: bool):
    normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    if is_train:
        return Compose([
            # RandomResizedCrop(n_px, scale=(0.9, 1.0), interpolation=Image.BICUBIC),   # default
            RandomResizedCrop(n_px, scale=(0.08, 1.0), interpolation=Image.BICUBIC),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
    else:
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])



def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit=True, is_train=False, pretrained=True):
    """Load a CLIP model
    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model
    jit : bool
        Whether to load the optimized JIT model (default) or more hackable non-JIT model.
    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if "clip" in name:
        name = name[:-5]
    if name in _MODELS:
        model_path = _download(_MODELS[name])
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    if not jit:
        try:
            # model = build_model(state_dict or model.state_dict())
            model = build_model(state_dict or model.state_dict()).to(device)
        except KeyError:
            sd = {k[7:]: v for k,v in state_dict["state_dict"].items()}
            model = build_model(sd).to(device)

        if str(device) == "cpu":
            model.float()
        return model, \
               _transform(model.visual.input_resolution, is_train=True), \
               _transform(model.visual.input_resolution, is_train=False)

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        graphs = [module.graph] if hasattr(module, "graph") else []
        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            graphs = [module.graph] if hasattr(module, "graph") else []
            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, \
           _transform(model.input_resolution.item(), is_train=True), \
           _transform(model.input_resolution.item(), is_train=False)


def tokenize(texts: Union[str, List[str]], context_length: int = 77) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<start_of_text>"]
    eot_token = _tokenizer.encoder["<end_of_text>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length: # Truncate
            tokens = tokens[:context_length]
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

def get_zeroshot_weight(clip_model, template, classnames):
    logit_scale = clip_model.logit_scale
    # device = args.device
    clip_model.eval()
    # clip_model.to(device)
    for params in list(clip_model.parameters()):
        device = params.device
        break

    print('Getting zeroshot weights.')
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = []
            for t in template:
                texts.append(t(classname))
            texts = tokenize(texts).to(device) # tokenize
            embeddings = clip_model.encode_text(texts) # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()
        
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    # classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return zeroshot_weights

def get_context_weights(clip_model, template, classnames):
    logit_scale = clip_model.logit_scale
    for params in list(clip_model.parameters()):
        device = params.device
        break
    clip_model.eval()
    # clip_model.to(device)

    print('Getting context weights.')
    with torch.no_grad():
        context_weights = []
        # TODO
        for t in tqdm(template):
            texts = []
            for classname in classnames:
                texts.append(t(classname))
            texts = tokenize(texts).to(device) # tokenize
            embeddings = clip_model.encode_text(texts) # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            context_weights.append(embeddings)

        context_weights = torch.stack(context_weights, dim=0).to(device)
        context_weights = torch.transpose(context_weights, 0, 2)

        context_weights *= logit_scale.exp()
        
        context_weights = context_weights.squeeze().float()
        context_weights = torch.transpose(context_weights, 0, 1)
    
    return context_weights

def get_kmeans_weights_classifier(clip_model, template, classnames, context_number):
    context_weights = get_context_weights(clip_model, template, classnames)
    device = context_weights.device
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=context_number, random_state=0).fit(context_weights.cpu().numpy())
    context_weights = torch.from_numpy(kmeans.cluster_centers_).to(device)
    return ClassificationHead(normalize=True, weights=context_weights)


def get_Gussianmixure_weight_clssifier(clip_model, template, classnames, context_number):
    context_weights = get_Gussanmixure_weight(clip_model, template, classnames, context_number)
    return ClassificationHead(normalize=True, weights=context_weights)

def get_Gussanmixure_weight(clip_model, template, classnames, context_number):
    torch.manual_seed(10)
    context_weights = get_context_weights(clip_model, template, classnames)
    device = context_weights.device
    # from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=5, random_state=0).fit(context_weights.cpu().numpy())
    context_weights = gmm.sample(context_number)[0]
    context_weights = torch.from_numpy(context_weights).to(device)
    # context_weights = torch.from_numpy(kmeans.cluster_centers_).to(device)
    return context_weights

def get_ILF_weights_classifier(clip_model, template, classnames, context_number):
    context_weights = get_context_weights(clip_model, template, classnames)
    device = context_weights.device
    context_weights = context_weights.cpu()
    from sklearn.ensemble import IsolationForest
    IlF = IsolationForest()
    IlF.fit(context_weights)
    scores = IlF.decision_function(context_weights)
    scores_=sorted(scores, reverse=True)
    labels = [1 if score > scores_[context_number] else 0 for score in scores]
    res_weigth=[]
    for i in range(len(labels)):
        if labels[i] == 1:
            res_weigth.append(context_weights[i].unsqueeze(0))
    context_weights = torch.cat(res_weigth, dim=0).to(device)
    return ClassificationHead(normalize=True, weights=context_weights)

def get_ILF_kmeans_weights_classifier(clip_model, template, classnames, context_number, ILF_number=30):
    context_weights = get_context_weights(clip_model, template, classnames)
    device = context_weights.device
    context_weights = context_weights.cpu()
    from sklearn.ensemble import IsolationForest
    IlF = IsolationForest()
    IlF.fit(context_weights)
    scores = IlF.decision_function(context_weights)
    scores_=sorted(scores, reverse=True)
    labels = [1 if score > scores_[ILF_number] else 0 for score in scores]
    res_weigth=[]
    for i in range(len(labels)):
        if labels[i] == 1:
            res_weigth.append(context_weights[i].unsqueeze(0))
    context_weights = torch.cat(res_weigth, dim=0)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=context_number, random_state=0).fit(context_weights.numpy())
    context_weights = torch.from_numpy(kmeans.cluster_centers_).to(device)
    # kmeans = KMeans(n_clusters=context_number, random_state=0).fit(context_weights.cpu().numpy())
    # context_weights = torch.from_numpy(kmeans.cluster_centers_).to(device)
    return ClassificationHead(normalize=True, weights=context_weights)

def get_context_random_weights(clip_model, template, classnames, context_number):
    context_weights = get_context_weights(clip_model, template, classnames)
    var = torch.var(context_weights, dim=0)
    mean = torch.mean(context_weights, dim=0)
    torch.manual_seed(10000)
    gaussian = torch.randn(context_number, mean.shape[0]).to(var.device)
    re_context_weight = gaussian * torch.sqrt(var) + mean

    return ClassificationHead(normalize=True, weights=re_context_weight)

def get_context_classifier(clip_model, template, classnames):

    context_weights = get_context_weights(clip_model, template, classnames)
    context_head = ClassificationHead(normalize=True, weights=context_weights)

    return context_head

def get_random_classifier(class_num, dim):
    weights = torch.randn(class_num, dim)
    weights = torch.transpose(weights, 0, 1)
    weights = weights / weights.norm(dim=0, keepdim=True)
    weights = torch.transpose(weights, 0, 1)
    return ClassificationHead(normalize=True, weights=weights)

class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone().float())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    @classmethod
    def load(cls, filename):
        print(f'Loading classification head from {filename}')
        return torch_load(filename)
    
def create_model():
    return ImageEncoder("ViT-B/32")

def get_CLIP_contrastive_logits(image_feature, text_embedding):
    text_embedding = text_embedding.to(image_feature.device)
    text_embedding = torch.div(text_embedding, text_embedding.norm(dim=-1, keepdim=True))
    image_feature = torch.div(image_feature, image_feature.norm(dim=-1, keepdim=True))
    
    logits_per_image = (100.0 * image_feature @ text_embedding.T).softmax(dim=-1)
    return logits_per_image
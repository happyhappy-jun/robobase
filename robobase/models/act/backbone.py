# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""

import torch
import torchvision
import time
from torch import nn
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter
from typing import List, Mapping, Optional, Any, OrderedDict
import timm
from timm.models.vision_transformer import VisionTransformer

from robobase.models.act.utils.resnet_film import resnet18 as resnet18_film
from robobase.models.act.utils.misc import NestedTensor, is_main_process
from robobase.models.act.position_encoding import build_position_encoding
import ssl


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than
    torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_layers: bool,
    ):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later
        # layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in
        # name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        # print(xs.shape)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
    ):
        # Stops "urllib.error.URLError: ... unable to get local issuer certificate"
        # When getting backbone
        ssl._create_default_https_context = ssl._create_unverified_context
        backbone = getattr(torchvision.models, name)(
            # replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(),
            # norm_layer=FrozenBatchNorm2d,
        )  # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

class ViTBackbone(nn.Module):
    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
    ):
        super().__init__()
        # Stops "urllib.error.URLError: ... unable to get local issuer certificate"
        ssl._create_default_https_context = ssl._create_unverified_context
        
        self.backbone = timm.create_model(name, pretrained=is_main_process(), num_classes=0)
        self.name = name
        
        # Freeze parameters if not training backbone
        if not train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.num_channels = self.backbone.num_features
        self.return_interm_layers = return_interm_layers
        
        # For ViT, we need to handle intermediate layers differently
        if return_interm_layers:
            # Create a dictionary to store intermediate outputs
            self.intermediate_outputs = {}
            
            # Register hooks for transformer blocks
            def hook_fn(name):
                def hook(module, input, output):
                    self.intermediate_outputs[name] = output
                return hook
            
            for i, block in enumerate(self.backbone.blocks):
                block.register_forward_hook(hook_fn(f'block_{i}'))
        
        # Set output shape for the encoder
        # ViT outputs [batch_size, num_patches + 1, hidden_dim]
        # We'll use the hidden_dim as the channel dimension
    
    @torch.no_grad()
    def forward(self, x):
        # Handle both Tensor and NestedTensor inputs
        if isinstance(x, NestedTensor):
            x = x.tensors
        
        # if x.shape[2] == 128 and x.shape[3] == 128:
            # if "336" in self.name:
                # x = F.interpolate(x, size=(336, 336), mode='bilinear', align_corners=False)
            # else:
                # x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        x = self.backbone.forward_features(x)  # [B, 197, 768]
        
        x = x[:, :1, :]  # Take cls token [B, 1, 768]
        x = x.permute(0, 2, 1)  # [B, 768, 1]
        x = x.reshape(x.shape[0], x.shape[1], 1, 1)  # [B, 768, 1, 1]
        
        return OrderedDict([("0", x)])

class CLIPBackbone(nn.Module):
    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
    ):
        super().__init__()
        # Stops "urllib.error.URLError: ... unable to get local issuer certificate"
        ssl._create_default_https_context = ssl._create_unverified_context
        
        self.backbone = timm.create_model(
            name,
            pretrained=is_main_process(),
            num_classes=0  # Remove classification head
        )
        
        # Freeze parameters if not training backbone
        if not train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.num_channels = self.backbone.num_features
        self.return_interm_layers = return_interm_layers
        
        # For ViT, we need to handle intermediate layers differently
        if return_interm_layers:
            # Create a dictionary to store intermediate outputs
            self.intermediate_outputs = {}
            
            # Register hooks for transformer blocks
            def hook_fn(name):
                def hook(module, input, output):
                    self.intermediate_outputs[name] = output
                return hook
            
            for i, block in enumerate(self.backbone.blocks):
                block.register_forward_hook(hook_fn(f'block_{i}'))
        
        # Set output shape for the encoder
        # ViT outputs [batch_size, num_patches + 1, hidden_dim]
        # We'll use the hidden_dim as the channel dimension
        self.output_shape = (self.num_channels,)
    
    def forward(self, x):
        # Handle both Tensor and NestedTensor inputs
        if isinstance(x, NestedTensor):
            x = x.tensors
        
        if x.shape[2] == 128 and x.shape[3] == 128:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Reset intermediate outputs if needed
        if self.return_interm_layers:
            self.intermediate_outputs.clear()
        
        # Forward pass through ViT
        with torch.no_grad():
            x = self.backbone.forward_features(x)  # [B, 197, 768]
        
        # Remove cls token and reshape to [B, C, H, W]
        # 197 = 1 (cls token) + 196 (14x14 patches)
        x = x[:, :1]  # Remove cls token [B, 196, 768]
        x = x.permute(0, 2, 1)  # [B, 768, 196]
        x = x.reshape(x.shape[0], x.shape[1], 1, 1)  # [B, 768, 14, 14]
        
        # Return intermediate outputs if requested
        if self.return_interm_layers:
            return self.intermediate_outputs
        
        # Return as OrderedDict to match ResNet format
        return OrderedDict([("0", x)])

class ResNetFilmBackbone(nn.Module):
    def __init__(
        self,
        embedding_name: str,
        pretrained: bool = True,
        film_config: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        self._pretrained = pretrained
        weights = "IMAGENET1K_V1" if pretrained else None
        if embedding_name in ("resnet18_film", "resnet18"):
            backbone = resnet18_film(
                weights=weights,
                film_config=film_config,
                pretrained=pretrained,
                norm_layer=FrozenBatchNorm2d,
            )
            embedding_dim = 512
        else:
            raise NotImplementedError

        self.resnet_film_model = backbone
        self._embedding_dim = embedding_dim
        self.resnet_film_model.fc = nn.Identity()
        self.resnet_film_model.avgpool = nn.Identity()

        self.num_channels = self._embedding_dim

        # FiLM config
        self.film_config = film_config
        if film_config is not None and film_config["use"]:
            film_models = []
            for layer_idx, num_blocks in enumerate(self.resnet_film_model.layers):
                if layer_idx in film_config["use_in_layers"]:
                    num_planes = self.resnet_film_model.film_planes[layer_idx]
                    film_model_layer = nn.Linear(
                        film_config["task_embedding_dim"], num_blocks * 2 * num_planes
                    )
                else:
                    film_model_layer = None
                film_models.append(film_model_layer)

            self.film_models = nn.ModuleList(film_models)

    def forward(
        self,
        x,
        texts: Optional[List[str]] = None,
        task_emb: Optional[torch.Tensor] = None,
        **kwargs
    ):
        film_outputs = None
        if self.film_config is not None and self.film_config["use"]:
            film_outputs = []
            for layer_idx, num_blocks in enumerate(self.resnet_film_model.layers):
                if self.film_config["use"] and self.film_models[layer_idx] is not None:
                    film_features = self.film_models[layer_idx](task_emb)
                else:
                    film_features = None
                film_outputs.append(film_features)
        return self.resnet_film_model(x, film_features=film_outputs, flatten=False)

    @property
    def embed_dim(self):
        return self._embedding_dim


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self._cached_pos = {}  # Cache for position encodings


    @torch.no_grad()
    def forward(self, tensor_list: NestedTensor, task_emb: Optional[Any] = None):
        start_time = time.time()
        if task_emb is not None:
            xs = self[0](tensor_list, task_emb=task_emb)
            # Make a dictionary out of the last layer outputs
            # since we don't have IntermediateLayerGetter
            xs = {"0": xs}
        else:
            xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            
            # Check if position encoding is in cache based on shape
            tensor_shape = (x.shape[-2], x.shape[-1])
            device = x.device
            dtype = x.dtype
            
            cache_key = f"{tensor_shape}_{device}_{dtype}"
            if cache_key not in self._cached_pos:
                # If not in cache, compute position encoding and cache it
                pos_encoding = self[1](x).to(x.dtype)
                self._cached_pos[cache_key] = pos_encoding
            else:
                # Retrieve from cache
                pos_encoding = self._cached_pos[cache_key]
                
                # Ensure cached encoding matches batch size
                if pos_encoding.shape[0] != x.shape[0]:
                    pos_encoding = pos_encoding[:x.shape[0]]
            
            pos.append(pos_encoding)

        end_time = time.time()
        # print(f"Time taken: {end_time - start_time} seconds")

        return out, pos


def build_backbone(
    hidden_dim, position_embedding, lr_backbone, masks, backbone, dilation
):
    position_embedding = build_position_encoding(hidden_dim, position_embedding)
    train_backbone = lr_backbone > 0
    print("Train backbone: ", train_backbone)
    return_interm_layers = masks
    
    if backbone == 'vit_base':
        backbone = ViTBackbone(
            name='vit_base_patch16_224',
            train_backbone=train_backbone,
            return_interm_layers=return_interm_layers,
            dilation=dilation,
        )
    elif backbone == 'clip_vit_l_336':
        backbone = ViTBackbone(
            name='vit_large_patch14_clip_336.openai',
            train_backbone=train_backbone,
            return_interm_layers=return_interm_layers,
            dilation=dilation,
        )
    else:
        backbone = Backbone(backbone, train_backbone, return_interm_layers, dilation)
    
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


def build_film_backbone(hidden_dim, position_embedding, backbone):
    position_embedding = build_position_encoding(hidden_dim, position_embedding)
    film_config = {
        "use": True,
        "use_in_layers": [1, 2, 3],
        "task_embedding_dim": hidden_dim,
        "film_planes": [64, 128, 256, 512],
    }

    backbone = ResNetFilmBackbone(backbone, film_config=film_config)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

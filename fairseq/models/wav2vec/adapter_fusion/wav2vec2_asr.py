# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
import contextlib
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from omegaconf import MISSING, II, open_dict
from typing import Any, Optional, List
from omegaconf import DictConfig
from fairseq.checkpoint_utils import prune_state_dict
import logging

from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.tasks import FairseqTask
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
)
from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
from fairseq.modules import LayerNorm, PositionalEmbedding, TransformerDecoderLayer

logger = logging.getLogger(__name__)


@dataclass
class Wav2Vec2AsrADConfig(FairseqDataclass):
    # szj
    adapter_cp: List[str] = field(
        default_factory=lambda: ['', ''],
        metadata={"help": "每个语种的adapter的checkpoint path"}
    )
    # szj added
    adapter_languages: List[str] = field(
        default_factory=lambda: ['107', '201'],
        metadata={
            "help": "每个adapter对应的语种"
        }
    )
    # szj
    freeze_fc: bool = field(
        default=True, metadata={"help": "whether freeze the final linear layer"}
    )
    # szj
    remove_old_fc: bool = field(
        default=False, metadata={"help": "whether remove the old output layer from checkpoint (例如bilingual节点在单语种上tune的时候要不要share vocab)"}
    )
    w2v_path: str = field(
        default=MISSING, metadata={"help": "path to wav2vec 2.0 model"}
    )
    no_pretrained_weights: bool = field(
        default=False, metadata={"help": "if true, does not load pretrained weights"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )
    dropout: float = field(
        default=0.0, metadata={"help": "dropout probability inside wav2vec 2.0 model"}
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside wav2vec 2.0 model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside wav2vec 2.0 model"
        },
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask (normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    freeze_finetune_updates: int = field(
        default=0, metadata={"help": "dont finetune wav2vec for this many updates"}
    )
    feature_grad_mult: float = field(
        default=0.0, metadata={"help": "reset feature grad mult in wav2vec 2.0 to this"}
    )
    layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a layer in wav2vec 2.0"}
    )
    normalize: bool = II("task.normalize")
    data: str = II("task.data")
    # this holds the loaded wav2vec args
    w2v_args: Any = None


@dataclass
class Wav2Vec2CtcADConfig(Wav2Vec2AsrADConfig):
    pass


@register_model("wav2vec_ctc_ad_fusion", dataclass=Wav2Vec2CtcADConfig)
class Wav2VecCtcAD(BaseFairseqModel):
    def __init__(self, cfg: Wav2Vec2CtcADConfig, w2v_encoder: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: Wav2Vec2CtcADConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = Wav2VecEncoderAD(cfg, task.target_dictionary)
        return cls(cfg, w2v_encoder)

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_logits(self, net_output):
        logits = net_output["encoder_out"]
        padding = net_output["encoder_padding_mask"]
        if padding is not None and padding.any():
            padding = padding.T
            logits[padding][...,0] = 0
            logits[padding][...,1:] = float('-inf')

        return logits

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x

    # szj, 处理输出层
    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg: Optional[DictConfig] = None,
        args: Optional[Namespace] = None,
    ):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """

        if model_cfg is None and args is not None:
            logger.warn("using 'args' is deprecated, please update your code to use dataclass config")
            model_cfg = convert_namespace_to_omegaconf(args).model

        self.upgrade_state_dict(state_dict)
        new_state_dict = prune_state_dict(state_dict, model_cfg)
        if self.cfg.remove_old_fc:
            new_state_dict.pop('w2v_encoder.proj.bias', None)  # 去掉输出层节点, 因为bilingual的节点和mono的节点不同
            new_state_dict.pop('w2v_encoder.proj.weight')

        # szj: 从每个语种的adapter checkpoint加载参数
        for lang, cp in zip(self.cfg.adapter_languages, self.cfg.adapter_cp):
            new_state_dict_adapter_only = {}
            new_state_dict_adapter = checkpoint_utils.load_checkpoint_to_cpu(cp)['model']

            for k in new_state_dict_adapter.keys():
                if '.adapter.' in k:
                    new_k = k.replace('adapter', f'adapter.adapters.{lang}')
                    new_state_dict_adapter_only[new_k] = new_state_dict_adapter[k]
                if '.adapter_sa.' in k:
                    new_k = k.replace('adapter_sa', f'adapter_sa.adapters.{lang}')
                    new_state_dict_adapter_only[new_k] = new_state_dict_adapter[k]

            new_state_dict = {**new_state_dict, **new_state_dict_adapter_only}

        return super().load_state_dict(new_state_dict, strict)

    # szj 加入adapterFusion的L2正则项, https://github.com/microsoft/NeuralSpeech/blob/c9561f3f5ac711cd13bde63272d49bec0867115e/AdapterASR/e2e_asr_adaptertransformer.py#L628
    def get_fusion_regularization_loss(self):
        reg_loss = 0.0
        fusion_reg_loss_weight = 0.01

        device = next(self.parameters()).device
        target = torch.zeros((self.w2v_encoder.w2v_model.encoder.embedding_dim, self.w2v_encoder.w2v_model.encoder.embedding_dim)).fill_diagonal_(1.0).to(device)

        for layer in self.w2v_encoder.w2v_model.encoder.layers:
            reg_loss = reg_loss + fusion_reg_loss_weight * (target - layer.adapter.sim_adapter.linear_v.weight).pow(2).sum()
            reg_loss += fusion_reg_loss_weight * (target - layer.adapter.sim_adapter.linear_v.weight).pow(2).sum()
        return reg_loss


    # 加入 Hou提出的guide loss  https://github.com/microsoft/NeuralSpeech/blob/23733d0c0fb57cf977ae064705ccb5d00e70e20e/AdapterASR/e2e_asr_adaptertransformer.py
    # 将注意力得分约束为[0, 1]这样的one hot向量
    # 必须要在fp时计算, 否则中间变量attn会被pytorch清空
    def get_fusion_guide_loss(self, language, net_output):
        adapter_attn_results = net_output['adapter_attn_results']
        adapter_sa_attn_results = net_output['adapter_sa_attn_results']

        device = next(self.parameters()).device
        # if language not in self.fusion_languages:
        #     return torch.tensor(0.0).to(device)

        guide_loss = torch.tensor(0.0).to(device)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        
        # lang_id = sorted(self.fusion_languages).index(language)
        lang_id = self.cfg.adapter_languages.index(language)
        # key = "_".join(self.fusion_languages)
        target = torch.tensor(lang_id).unsqueeze(0).to(device)

        for attn in adapter_attn_results + adapter_sa_attn_results:
            if attn is not None:
                logits = attn.mean(axis=(0, 1)).unsqueeze(0) # (batch, time, n_adapters)
                guide_loss = guide_loss + loss_fn(logits.exp(), target)
        return guide_loss


class Wav2VecEncoderAD(FairseqEncoder):
    def __init__(self, cfg: Wav2Vec2AsrADConfig, tgt_dict=None):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )

        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task)
        w2v_args.model['_name'] = 'wav2vec2_ad_fusion'
        with open_dict(w2v_args):  # 对于omegaconf, 新增参数的写法, refer: https://stackoverflow.com/questions/66295334/create-a-new-key-in-hydra-dictconfig-from-python-file
            w2v_args.model.adapter_languages = cfg.adapter_languages
        model = task.build_model(w2v_args.model)

        if state is not None and not cfg.no_pretrained_weights:
            # szj True --> False
            model.load_state_dict(state["model"], strict=False)

        model.remove_pretraining_modules()

        super().__init__(task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim
        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        if tgt_dict is not None:
            self.proj = Linear(d, len(tgt_dict))
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            self.proj = Linear(d, cfg.decoder_embed_dim)
        else:
            self.proj = None

        # szj 冻结除了adapterFusion的参数
        for name, param in self.w2v_model.named_parameters():
            if param.requires_grad and 'sim_adapter' not in name:
                param.requires_grad = False
        if cfg.freeze_fc:
            for name, param in self.proj.named_parameters():
                if param.requires_grad and 'sim_adapter' not in name:
                    param.requires_grad = False

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, **kwargs):

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask, adapter_attn_results, adapter_sa_attn_results = self.w2v_model.extract_features(**w2v_args)

            if tbc:
                # B x T x C -> T x B x C
                x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.proj:
            x = self.proj(x)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
            "adapter_attn_results": adapter_attn_results,
            "adapter_sa_attn_results": adapter_sa_attn_results,  # szj
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict



def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import imp
from logging import raiseExceptions
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor, nn
from torch.nn import Parameter

# szj
from fairseq.models.wav2vec.prefix_tune_langemb_langspec_adapter.attention import multi_head_attention_forward


class PrefixTuningLangEmbLangspec(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        input_size: int,
        # config: PrefixTuningConfig,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.input_size = input_size
        self.n_embd_per_head = self.input_size // self.n_heads

        # 参考配置：https://github.com/Adapter-Hub/adapter-transformers/blob/d8bcf37b4d3f85b81a0394b06db7b4e3f8a8f6c8/src/transformers/adapters/configuration.py
        # https://adapterhub.ml/blog/2022/03/adapter-transformers-v3-unifying-efficient-fine-tuning/#prefix-tuning
        self.prefix_length = 1 # 30
        self.vocab_size = 5  # 4个语种 + 特殊符号（处理空样本）
        self.bottleneck_size = 800
        # self.non_linearity = torch.tanh()
        self.dropout = 0.0

        # self.input_tokens = torch.arange(self.prefix_length).long()
        self.wte = nn.Embedding(self.vocab_size, self.input_size)
        # self.control_trans = nn.Sequential(
        #     nn.Linear(self.input_size, self.bottleneck_size),
        #     nn.Tanh(),
        #     nn.Linear(self.bottleneck_size, self.n_layers * 2 * self.input_size),
        # )

        # 每个语种一套参数
        self.control_trans_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_size, self.bottleneck_size),
                nn.Tanh(),
                nn.Linear(self.bottleneck_size, self.n_layers * 2 * self.input_size),
            ) for _ in range(5)
        ])

        self.dropout = nn.Dropout(self.dropout)


    def forward(self, batch_size, language_id):
        device = next(self.parameters()).device
        # input_tokens = self.input_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        input_tokens = language_id.unsqueeze(-1).to(device)  # (b x 1)
        embs = self.wte(input_tokens)  # (b x 1 x input_size)

        # key_values = self.control_trans(embs)  # batch_size x prefix_length x n_layers*2*input_size
        # szj 每个语种一套参数
        output = []
        for c in self.control_trans_list:
            output.append(c(embs))  # b x 1 x n_layers*2*input_size
        output = torch.stack(output)  # n_languages x b x 1 x n_layers*2*input_size
        weight = F.one_hot(language_id.long(), num_classes=5).transpose(0, 1).contiguous()  # 5 x b

        key_values = weight.unsqueeze(-1).unsqueeze(-1) * output  # n_languages x b x 1 x n_layers*2*input_size
        key_values = key_values.sum(0)  # b x 1 x n_layers*2*input_size

        key_values = key_values.view(
            batch_size, self.prefix_length, self.n_layers * 2, self.n_heads, self.n_embd_per_head
        )  # *2 for key and value
        
        key_values = self.dropout(key_values)
        # n_layers * (2 x batch_size x n_heads x prefix_length x n_embd_per_head)
        key_values = key_values.permute(2, 0, 3, 1, 4).split(2)

        return key_values


# szj
class PrefixTuningLangEmb(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        input_size: int,
        # config: PrefixTuningConfig,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.input_size = input_size
        self.n_embd_per_head = self.input_size // self.n_heads

        # 参考配置：https://github.com/Adapter-Hub/adapter-transformers/blob/d8bcf37b4d3f85b81a0394b06db7b4e3f8a8f6c8/src/transformers/adapters/configuration.py
        # https://adapterhub.ml/blog/2022/03/adapter-transformers-v3-unifying-efficient-fine-tuning/#prefix-tuning
        self.prefix_length = 1 # 30
        self.vocab_size = 5  # 4个语种 + 特殊符号（处理空样本）
        self.bottleneck_size = 800
        # self.non_linearity = torch.tanh()
        self.dropout = 0.0

        # self.input_tokens = torch.arange(self.prefix_length).long()
        self.wte = nn.Embedding(self.vocab_size, self.input_size)
        self.control_trans = nn.Sequential(
            nn.Linear(self.input_size, self.bottleneck_size),
            nn.Tanh(),
            nn.Linear(self.bottleneck_size, self.n_layers * 2 * self.input_size),
        )


        self.dropout = nn.Dropout(self.dropout)

    def eject(self):
        device = next(self.parameters()).device
        input_tokens = self.input_tokens.unsqueeze(0).expand(1, -1).to(device)
        embs = self.wte(input_tokens)
        key_values = self.control_trans(embs)  # batch_size x prefix_length x n_layers*2*input_size
        key_values = key_values.view(
            self.prefix_length * self.n_layers * 2 * self.input_size
        )  # *2 for key and value

        return key_values

    def forward(self, batch_size, language_id):
        device = next(self.parameters()).device
        # input_tokens = self.input_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        # TODO prefix length为1怎么解决？
        input_tokens = language_id.unsqueeze(-1).unsqueeze(-1).to(device)  # (b x 1 x 1)
        embs = self.wte(input_tokens)  # (b x 1 x input_size)
        key_values = self.control_trans(embs)  # batch_size x prefix_length x n_layers*2*input_size
        key_values = key_values.view(
            batch_size, self.prefix_length, self.n_layers * 2, self.n_heads, self.n_embd_per_head
        )  # *2 for key and value
        
        key_values = self.dropout(key_values)
        # n_layers * (2 x batch_size x n_heads x prefix_length x n_embd_per_head)
        key_values = key_values.permute(2, 0, 3, 1, 4).split(2)

        return key_values


# szj
# copied from https://github.com/Adapter-Hub/adapter-transformers/blob/master/src/transformers/adapters/prefix_tuning.py
class PrefixTuning(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        input_size: int,
        # config: PrefixTuningConfig,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.input_size = input_size
        self.n_embd_per_head = self.input_size // self.n_heads

        # 参考配置：https://github.com/Adapter-Hub/adapter-transformers/blob/d8bcf37b4d3f85b81a0394b06db7b4e3f8a8f6c8/src/transformers/adapters/configuration.py
        # https://adapterhub.ml/blog/2022/03/adapter-transformers-v3-unifying-efficient-fine-tuning/#prefix-tuning
        self.prefix_length = 30
        self.bottleneck_size = 800
        # self.non_linearity = torch.tanh()
        self.dropout = 0.0

        self.input_tokens = torch.arange(self.prefix_length).long()
        self.wte = nn.Embedding(self.prefix_length, self.input_size)
        self.control_trans = nn.Sequential(
            nn.Linear(self.input_size, self.bottleneck_size),
            nn.Tanh(),
            nn.Linear(self.bottleneck_size, self.n_layers * 2 * self.input_size),
        )
        self.dropout = nn.Dropout(self.dropout)

    def eject(self):
        device = next(self.parameters()).device
        input_tokens = self.input_tokens.unsqueeze(0).expand(1, -1).to(device)
        embs = self.wte(input_tokens)
        key_values = self.control_trans(embs)  # batch_size x prefix_length x n_layers*2*input_size
        key_values = key_values.view(
            self.prefix_length * self.n_layers * 2 * self.input_size
        )  # *2 for key and value

        return key_values

    def forward(self, batch_size):
        device = next(self.parameters()).device
        input_tokens = self.input_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        embs = self.wte(input_tokens)
        key_values = self.control_trans(embs)  # batch_size x prefix_length x n_layers*2*input_size
        key_values = key_values.view(
            batch_size, self.prefix_length, self.n_layers * 2, self.n_heads, self.n_embd_per_head
        )  # *2 for key and value
        key_values = self.dropout(key_values)
        # n_layers * (2 x batch_size x n_heads x prefix_length x n_embd_per_head)
        key_values = key_values.permute(2, 0, 3, 1, 4).split(2)

        return key_values


# szj copied from https://github.com/Adapter-Hub/adapter-transformers/blob/master/src/transformers/adapters/prefix_tuning.py
# 直接生成一组权重
class FlatPrefixTuning(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        input_size: int,
        # config: PrefixTuningConfig,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.input_size = input_size
        self.n_embd_per_head = self.input_size // self.n_heads
        # self.config = config

        # szj added config
        self.prefix_length = 200
        self.dropout = 0.0

        self.control_trans = nn.Parameter(torch.randn(self.prefix_length * self.n_layers * 2 * self.input_size))

        self.dropout = nn.Dropout(self.dropout)

    def forward(self, batch_size):
        device = next(self.parameters()).device
        key_values = (
            self.control_trans.unsqueeze(0)
            .expand(batch_size, -1)
            .view(batch_size, self.prefix_length, self.n_layers * 2, self.n_heads, self.n_embd_per_head)
            .to(device)
        )  # *2 for key and value
        key_values = self.dropout(key_values)
        # n_layers * (2 x batch_size x n_heads x prefix_length x n_embd_per_head)
        key_values = key_values.permute(2, 0, 3, 1, 4).split(2)

        return key_values


@with_incremental_state
class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        self.k_proj = quant_noise(
            nn.Linear(self.kdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = quant_noise(
            nn.Linear(self.vdim, embed_dim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = quant_noise(
            nn.Linear(embed_dim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False
        # szj 增加prefix-tuning模块，用于取得prefix
        self.prefix_tune = PrefixTuningLangEmbLangspec(n_layers=1, n_heads=self.num_heads, input_size=self.embed_dim)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        language_id: Tensor = None,
        prefix_keys_values: Tensor = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if (
            not self.onnx_trace
            and not is_tpu  # don't use PyTorch version on TPUs
            and incremental_state is None
            and not static_kv
            # A workaround for quantization to work. Otherwise JIT compilation
            # treats bias in linear module as method.
            and not torch.jit.is_scripting()
        ):
            assert key is not None and value is not None

            # szj 对key value进行处理
            prefix_keys_values = self.prefix_tune(bsz, language_id)[0]  # (2 x bsz x n_heads x prefix_length x n_embd_per_head)
            prefix_keys = prefix_keys_values[0, ...]  # (bsz, n_heads, prefix_length, n_embd_per_head)
            prefix_values= prefix_keys_values[1, ...]  # (bsz, n_heads, prefix_length, n_embd_per_head)

            return multi_head_attention_forward(
                prefix_keys,
                prefix_values,
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout_module.p,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training or self.dropout_module.apply_during_inference,
                key_padding_mask,
                need_weights,
                attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )

    @staticmethod
    def _append_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), key_padding_mask.float()], dim=1
            )
        # During incremental decoding, as the padding token enters and
        # leaves the frame, there will be a time when prev or current
        # is None
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - prev_key_padding_mask.size(1)),
                device=prev_key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [prev_key_padding_mask.float(), filler.float()], dim=1
            )
        elif key_padding_mask is not None:
            filler = torch.zeros(
                (batch_size, src_len - key_padding_mask.size(1)),
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat(
                [filler.float(), key_padding_mask.float()], dim=1
            )
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention and input_buffer_k.size(
                        0
                    ) == new_order.size(0):
                        break
                    input_buffer[k] = input_buffer_k.index_select(0, new_order)
            incremental_state = self._set_input_buffer(incremental_state, input_buffer)
        return incremental_state

    def _get_input_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ) -> Dict[str, Optional[Tensor]]:
        result = self.get_incremental_state(incremental_state, "attn_state")
        if result is not None:
            return result
        else:
            empty_result: Dict[str, Optional[Tensor]] = {}
            return empty_result

    def _set_input_buffer(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        return self.set_incremental_state(incremental_state, "attn_state", buffer)

    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value

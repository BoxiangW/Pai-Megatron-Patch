# Copyright (c) 2024 Alibaba PAI and Nvidia Megatron-LM Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union
import math
import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.utils import divide
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron_patch.model.deepseek_v2.yarn_rotary_pos_embedding import DeepseekV2YarnRotaryEmbedding, \
    apply_rotary_pos_emb, yarn_get_mscale

@dataclass
class SelfAttentionSubmodules:
    linear_q_proj: Union[ModuleSpec, type] = None
    linear_q_a_proj: Union[ModuleSpec, type] = None
    linear_q_b_proj: Union[ModuleSpec, type] = None
    linear_kv_a_proj_with_mqa: Union[ModuleSpec, type] = None
    linear_kv_b_proj: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_a_layernorm: Union[ModuleSpec, type] = None
    kv_a_layernorm: Union[ModuleSpec, type] = None


class Attention(MegatronModule, ABC):
    """Attention layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Union[SelfAttentionSubmodules],
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
    ):
        super().__init__(config=config)

        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type

        # For normal attention without groups, num_query_groups == num_attention_heads,
        # so these two will be the same
        self.query_projection_size = self.config.v_head_dim * self.config.num_attention_heads
        self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups

        # Per attention head and per partition values.
        self.world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = divide(
            self.query_projection_size, self.config.num_attention_heads
        )
        self.num_attention_heads_per_partition = divide(
            self.config.num_attention_heads, self.world_size
        )
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, self.world_size)

        self.q_head_dim = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim

        mscale = yarn_get_mscale(40, 0.707)
        self.softmax_scale = 1 / math.sqrt(self.q_head_dim) * mscale * mscale

        self.core_attention = build_module(
            submodules.core_attention,
            config=self.config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
            softmax_scale=self.softmax_scale,
        )

        self.checkpoint_core_attention = self.config.recompute_granularity == 'selective'

        # print("self.query_projection_size,", self.query_projection_size)
        
        # Output.
        self.linear_proj = build_module(
            submodules.linear_proj,
            self.query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name='proj',
        )

        kwargs = {
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
        }

        self.rotary_pos_emb = DeepseekV2YarnRotaryEmbedding(
            self.config.qk_rope_head_dim,
            base=self.config.rotary_base,
            max_position_embeddings=self.config.max_position_embeddings,
            scaling_factor=self.config.rotary_scaling_factor,
            **kwargs,
        )

    def _checkpointed_attention_forward(
        self,
        query,
        key,
        value,
        attention_mask,
        rotary_pos_emb=None,
        attn_mask_type=None,
        packed_seq_params=None,
    ):
        """Forward method with selective activation checkpointing."""

        def custom_forward(*inputs):
            query = inputs[0]
            key = inputs[1]
            value = inputs[2]
            attention_mask = inputs[3]
            attn_mask_type = inputs[5]
            attn_mask_type = AttnMaskType(attn_mask_type.item())
            output_ = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
            return output_

        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type
        attn_mask_type = torch.tensor([attn_mask_type.value], dtype=torch.int)
        hidden_states = tensor_parallel.checkpoint(
            custom_forward,
            False,
            query,
            key,
            value,
            attention_mask,
            rotary_pos_emb,
            attn_mask_type,
        )

        return hidden_states

    @abstractmethod
    def get_query_key_value_tensors(self, hidden_states, key_value_states, position_ids):
        """
        This method needs to be implemented based on whether the derived class
        is "self-attn" or "cross-attn".
        """

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
        position_ids=None,
    ):
        # hidden_states: [sq, b, h]
        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        # query: [96, 1, 16, 128], key:[96, 1, 16, 128], value:[96, 1, 16, 128]
        query, key, value = self.get_query_key_value_tensors(
            hidden_states, key_value_states, position_ids
        )

        # ==================================
        # core attention computation
        # ==================================
        # Need corresponding TE change
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query, key, value, attention_mask, packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                packed_seq_params=packed_seq_params,
                attn_mask_type=self.attn_mask_type,
            )

        if packed_seq_params is not None:
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # =================
        # Output. [sq, b, h]
        # =================
        # print("core_attn_out.shape", core_attn_out.shape)        
        core_attn_out = core_attn_out[:, :, : self.query_projection_size]
        output, bias = self.linear_proj(core_attn_out)

        return output, bias


class SelfAttention(Attention):
    """Self-attention layer class

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
        )

        if self.config.q_lora_rank is None:

            self.linear_q_proj = build_module(
                submodules.linear_q_proj,
                self.config.hidden_size,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
            )

        else:

            self.linear_q_a_proj = build_module(
                submodules.linear_q_a_proj,
                self.config.hidden_size,
                self.config.q_lora_rank,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
            )

            self.linear_q_b_proj = build_module(
                submodules.linear_q_b_proj,
                self.config.q_lora_rank//self.world_size,
                self.config.num_attention_heads * self.q_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
            )

        self.linear_kv_a_proj_with_mqa = build_module(
            submodules.linear_kv_a_proj_with_mqa,
            self.config.hidden_size,
            (self.config.kv_lora_rank + self.config.qk_rope_head_dim)*self.world_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        )

        self.linear_kv_b_proj = build_module(
            submodules.linear_kv_b_proj,
            self.config.kv_lora_rank,
            self.config.num_attention_heads * (self.q_head_dim - self.config.qk_rope_head_dim + self.config.v_head_dim),
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        )

        if self.config.q_lora_rank is not None:

            self.q_a_layernorm = build_module(
                submodules.q_a_layernorm,
                hidden_size=self.config.q_lora_rank//self.world_size,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )

        self.kv_a_layernorm = build_module(
            submodules.kv_a_layernorm,
            hidden_size=self.config.kv_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None, position_ids=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        # import pdb; pdb.set_trace()
        q_len, bsz, _ = hidden_states.size()     
        if self.config.q_lora_rank is not None:
            q_compressed, _ = self.linear_q_a_proj(hidden_states)
            q_compressed = self.q_a_layernorm(q_compressed)
            q, _ = self.linear_q_b_proj(q_compressed)
        else:
            # hidden_states:[24, 1, 2048], q: [48, 1, 1536]
            q, _ = self.linear_q_proj(hidden_states)
        # print('q.shape', q.shape)
        # q: [48, 1, 8, 192]
        q = q.view(q_len, bsz, self.num_attention_heads_per_partition, self.q_head_dim)

        # q: [48, 1, 8, 128], q_pos_emb: [48, 1, 8, 64]
        q, q_pos_emb = torch.split(
            q, [self.config.qk_nope_head_dim, self.config.qk_rope_head_dim], dim=-1
        )

        # kv_combined: [48, 1, 576]
        kv_combined, _ = self.linear_kv_a_proj_with_mqa(hidden_states)
        
        # print(len(kv_combined))

        # kv_compressed:[48, 1, 512], k_pos_emb: [48, 1, 64]
        kv_compressed, k_pos_emb = torch.split(
            kv_combined, [self.config.kv_lora_rank, self.config.qk_rope_head_dim], dim=-1
        )

        # kv: [48, 1, 2048]
        # print("kv_compressed.shape", kv_compressed.shape)
        kv, _ = self.linear_kv_b_proj(self.kv_a_layernorm(kv_compressed))

        # kv: [48, 1, 8, 256]
        # print("kv.shape", kv.shape)
        kv = kv.view(
            q_len,
            bsz,
            self.num_attention_heads_per_partition,
            self.config.qk_nope_head_dim + self.config.v_head_dim,
        )

        # k: [48, 1, 8, 128], value: [48, 1, 8, 128]
        k, value = torch.split(kv, [self.config.qk_nope_head_dim, self.config.v_head_dim], dim=-1)
        # print("k.shape", k.shape)
        # print("value.shape", value.shape)

        # value: [48, 1, 8, 128] -> [1, 8, 48, 128]
        value = value.transpose(0, 1).transpose(1, 2)
        kv_seq_len = value.shape[-2]

        # cos: [48, 64], sin:[48, 64]
        cos, sin = self.rotary_pos_emb(value, seq_len=kv_seq_len)

        # [48, 1, 8, 64] -> [1, 8, 48, 64]
        q_pos_emb = q_pos_emb.transpose(0, 1).transpose(1, 2)
        # [48, 1, 64] -> [1, 48, 64]
        k_pos_emb = k_pos_emb.transpose(0, 1)
        # [1, 1, 48, 64]
        k_pos_emb = k_pos_emb.reshape(bsz, q_len, 1, -1).transpose(1, 2)

        # q_pos_emb: [1, 8, 48, 64], k_pos_emb:[1, 1, 48, 64]
        q_pos_emb, k_pos_emb = apply_rotary_pos_emb(
            q_pos_emb, k_pos_emb, cos, sin, position_ids[:, :kv_seq_len]
        )

        # query: [1, 8, 48, 192]
        query = k_pos_emb.new_empty(
            bsz, self.num_attention_heads_per_partition, q_len, self.q_head_dim
        )

        # q: [48, 1, 8, 128] -> [1, 8, 48, 128]
        q = q.transpose(0, 1).transpose(1, 2)

        query[:, :, :, : self.config.qk_nope_head_dim] = q
        query[:, :, :, self.config.qk_nope_head_dim :] = q_pos_emb

        # key: [1, 8, 48, 192]
        key = k_pos_emb.new_empty(
            bsz, self.num_attention_heads_per_partition, q_len, self.q_head_dim
        )
        # k: [48, 1, 8, 128] -> [1, 8, 48, 128] 
        k = k.transpose(0, 1).transpose(1, 2)
        # print("+k.shape", k.shape)
        # print("+key.shape", key.shape)
        # print("+k_pos_emb.shape", k_pos_emb.shape)
        key[:, :, :, : self.config.qk_nope_head_dim] = k
        key[:, :, :, self.config.qk_nope_head_dim :] = k_pos_emb
        
        # pad_value: [1, 8, 48, 192]
        
        pad_value = k_pos_emb.new_zeros(
            bsz, self.num_attention_heads_per_partition, q_len, self.q_head_dim
        )
        # print(pad_value.shape)
        pad_value[:, :, :, : self.config.qk_nope_head_dim] = value
        # print(pad_value)
        
        # bhsd -> sbhd
        query = query.transpose(0, 2).transpose(1, 2).contiguous()
        key = key.transpose(0, 2).transpose(1, 2).contiguous()
        pad_value = pad_value.transpose(0, 2).transpose(1, 2).contiguous()
        # print(query.is_contiguous(), key.is_contiguous(), pad_value.is_contiguous())
        
        # print("shape", query.shape)

        return query, key, pad_value


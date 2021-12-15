# -*- coding: utf-8 -*-
import math
import random
import torch
from torch import nn
from typing import Optional, Tuple
from model.ModelOutput import ModelOutput

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

ACT2FN = {
    "relu": nn.functional.relu,
    "gelu": gelu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
}

class DecoderOutput(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).
    """
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


class PlanPositionalEmbedding(nn.Embedding):
    """
    Base class for positional embeddings of plans.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions + self.offset)


class MultiHeadAttention(nn.Module):
    """
    Multi-headed attention for the encoder-decoder framework.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        init_std: float = 0.01,
        is_decoder: bool = False,
        bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.init_std = init_std
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _init_all_weights(self):
        self.k_proj.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.k_proj.bias is not None:
            self.k_proj.bias.data.zero_()
        self.v_proj.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.v_proj.bias is not None:
            self.v_proj.bias.data.zero_()
        self.q_proj.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.q_proj.bias is not None:
            self.q_proj.bias.data.zero_()
        self.out_proj.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.out_proj.bias is not None:
            self.out_proj.bias.data.zero_()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # if `key_value_states` are provided, this layer is used as a cross-attention layer for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class KTMutualAttention(nn.Module):
    """
    Knowledge-Target Mutual Attention with multi-heads.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        init_std: float = 0.01,
        is_decoder: bool = False,
        bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.init_std = init_std
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.weighted_k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.weighted_q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _init_all_weights(self):
        self.k_proj.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.k_proj.bias is not None:
                self.k_proj.bias.data.zero_()
        self.v_proj.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.v_proj.bias is not None:
                self.v_proj.bias.data.zero_()
        self.q_proj.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.q_proj.bias is not None:
                self.q_proj.bias.data.zero_()
        self.out_proj.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.out_proj.bias is not None:
                self.out_proj.bias.data.zero_()
        self.weighted_k_proj.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.weighted_k_proj.bias is not None:
                self.weighted_k_proj.bias.data.zero_()
        self.weighted_q_proj.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.weighted_q_proj.bias is not None:
                self.weighted_q_proj.bias.data.zero_()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor,
        target_states: torch.Tensor,
        target_mask: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()
        _, target_len, _ = target_states.size()

        query_states = self.q_proj(hidden_states) * self.scaling
        target_query_states = self.weighted_q_proj(key_value_states) * self.scaling

        if is_cross_attention and past_key_value is not None:
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
            target_key_states = self._shape(self.weighted_k_proj(target_states), -1, bsz)
        elif past_key_value is not None:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )
        
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # compute target-aware weights
        target_query_states = self._shape(target_query_states, src_len, bsz).view(*proj_shape)
        target_key_states = target_key_states.view(*proj_shape)
        target_attn_weights = torch.bmm(target_query_states, target_key_states.transpose(1, 2))
        if target_attn_weights.size() != (bsz * self.num_heads, src_len, target_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, src_len, target_len)}, but is {target_attn_weights.size()}"
            )
        
        if target_mask.size() != (bsz, 1, src_len, target_len):
            raise ValueError(
                f"Target Attention mask should be of size {(bsz, 1, src_len, target_len)}, but is {target_mask.size()}"
            )
        target_mask = target_mask.repeat(1, self.num_heads, 1, 1)
        # mean pooling
        weights = torch.sum(target_attn_weights.view(bsz, self.num_heads, src_len, target_len) * target_mask,
            dim=-1) * 1.0 / torch.sum(target_mask, dim=-1)
        weights = weights.view(bsz * self.num_heads, src_len).unsqueeze(1).repeat(1, tgt_len, 1)
        
        # re-weight attention weights
        attn_weights = nn.functional.softmax(attn_weights * weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class InformationFusion(nn.Module):
    """
    Information Fusion Layer.
    """
    def __init__(self, d_model, init_std=0.01):
        super().__init__()
        self.init_std = init_std
        self.gate_layer1 = nn.Linear(d_model*2, 1, bias=True)
        self.gate_layer2 = nn.Linear(d_model*2, 1, bias=True)
        self.sigmod = nn.Sigmoid()
    
    def _init_all_weights(self):
        self.gate_layer1.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.gate_layer1.bias is not None:
            self.gate_layer1.bias.data.zero_()
        self.gate_layer2.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.gate_layer2.bias is not None:
            self.gate_layer2.bias.data.zero_()

    def forward(self, k_states, h_states, u_states):
        gate1 = self.sigmod(self.gate_layer1(torch.cat([k_states, h_states],  -1)))
        states = gate1 * k_states + (1 - gate1) * h_states
        gate2 = self.sigmod(self.gate_layer2(torch.cat([u_states, states],  -1)))
        f_states = gate2 * u_states + (1 - gate2) * states
        return f_states


class DecoderLayer(nn.Module):
    """
    Decoder Layer.
    """
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.embed_dim
        self.init_std = args.init_std
        self.dropout = args.dropout
        assert args.activation_function in ("relu", "gelu", "tanh", "sigmoid")
        self.activation_fn = ACT2FN[args.activation_function]
        self.activation_dropout = args.activation_dropout

        # masked multi-head self attentios for querying different parts
        self.kg_self_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            init_std=args.init_std,
            is_decoder=True
        )
        self.up_self_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            init_std=args.init_std,
            is_decoder=True
        )
        self.hs_self_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            init_std=args.init_std,
            is_decoder=True
        )
        self.kg_self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.up_self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.hs_self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # knowledge-target mutual attention layer
        self.kt_mutual_attn = KTMutualAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            init_std=args.init_std,
            is_decoder=True
        )
        # user preference cross attention layer
        self.up_cross_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            init_std=args.init_std,
            is_decoder=True
        )
        # context-aware cross attention layer
        self.ca_cross_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            init_std=args.init_std,
            is_decoder=True
        )
        self.kt_mutual_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.up_cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.ca_cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # information fusion layer
        self.fusion_layer = InformationFusion(d_model=self.embed_dim, init_std=self.init_std)
        self.fc1 = nn.Linear(self.embed_dim, args.decoder_ffn_dim)
        self.fc2 = nn.Linear(args.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
    
    def _init_all_weights(self):
        self.fc1.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.fc1.bias is not None:
            self.fc1.bias.data.zero_()
        self.fc2.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.fc2.bias is not None:
            self.fc2.bias.data.zero_()
        self.kg_self_attn._init_all_weights()
        self.up_self_attn._init_all_weights()
        self.hs_self_attn._init_all_weights()
        self.kt_mutual_attn._init_all_weights()
        self.up_cross_attn._init_all_weights()
        self.ca_cross_attn._init_all_weights()
        self.fusion_layer._init_all_weights()

    def forward(
        self,
        hidden_states: torch.Tensor,                                # (batch_size, seq_len, hidden_size)
        attention_mask: torch.Tensor = None,                        # (batch_size, 1, tgt_len, src_len)
        target_states: torch.Tensor=None,                           # (batch_size, seq_len, hidden_size)
        target_mask: torch.Tensor = None,                           # (batch_size, 1, tgt_len, src_len)
        kg_encoder_hidden_states: torch.Tensor = None,              # (batch_size, seq_len, hidden_size)
        kg_encoder_attention_mask: Optional[torch. Tensor] = None,  # (batch_size, 1, tgt_len, src_len)
        up_encoder_hidden_states: torch.Tensor = None,              # (batch_size, seq_len, hidden_size)
        up_encoder_attention_mask: Optional[torch.Tensor] = None,   # (batch_size, 1, tgt_len, src_len)
        hs_encoder_hidden_states: torch.Tensor = None,              # (batch_size, seq_len, hidden_size)
        hs_encoder_attention_mask: Optional[torch.Tensor] = None,   # (batch_size, 1, tgt_len, src_len)
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False
    ):
        residual = hidden_states
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # masked multi-head self-attention for knowledge
        kg_hidden_states, _, kg_present_key_value = self.kg_self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        kg_hidden_states = nn.functional.dropout(kg_hidden_states, p=self.dropout, training=self.training)
        kg_hidden_states = residual + kg_hidden_states
        kg_hidden_states = self.kg_self_attn_layer_norm(kg_hidden_states)
        # masked multi-head self-attention for user memory
        up_hidden_states, _, up_present_key_value = self.up_self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        up_hidden_states = nn.functional.dropout(up_hidden_states, p=self.dropout, training=self.training)
        up_hidden_states = residual + up_hidden_states
        up_hidden_states = self.up_self_attn_layer_norm(up_hidden_states)
        # masked multi-head self-attention for conversation history
        hs_hidden_states, _, hs_present_key_value = self.hs_self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hs_hidden_states = nn.functional.dropout(hs_hidden_states, p=self.dropout, training=self.training)
        hs_hidden_states = residual + hs_hidden_states
        hs_hidden_states = self.hs_self_attn_layer_norm(hs_hidden_states)

        ######### Cross-Attention Block ########
        kg_cross_attn_present_key_value = None
        if kg_encoder_hidden_states is not None:
            residual = kg_hidden_states
            kg_cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # knowledge-target mutual attention
            kg_hidden_states, _, kg_cross_attn_present_key_value = self.kt_mutual_attn(
                hidden_states=kg_hidden_states,
                key_value_states=kg_encoder_hidden_states,
                target_states=target_states,
                target_mask=target_mask,
                attention_mask=kg_encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=kg_cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            kg_hidden_states = nn.functional.dropout(kg_hidden_states, p=self.dropout, training=self.training)
            kg_hidden_states = residual + kg_hidden_states
            kg_hidden_states = self.kt_mutual_attn_layer_norm(kg_hidden_states)
            kg_present_key_value = kg_present_key_value + kg_cross_attn_present_key_value

        up_cross_attn_present_key_value = None
        if up_encoder_hidden_states is not None:
            residual = up_hidden_states
            up_cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # user preference cross attention
            up_hidden_states, _, up_cross_attn_present_key_value = self.up_cross_attn(
                hidden_states=up_hidden_states,
                key_value_states=up_encoder_hidden_states,
                attention_mask=up_encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=up_cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            up_hidden_states = nn.functional.dropout(up_hidden_states, p=self.dropout, training=self.training)
            up_hidden_states = residual + up_hidden_states
            up_hidden_states = self.up_cross_attn_layer_norm(up_hidden_states)
            up_present_key_value = up_present_key_value + up_cross_attn_present_key_value

        hs_cross_attn_present_key_value = None
        if hs_encoder_hidden_states is not None:
            residual = hs_hidden_states
            hs_cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # context-aware cross attention
            hs_hidden_states, _, hs_cross_attn_present_key_value = self.ca_cross_attn(
                hidden_states=hs_hidden_states,
                key_value_states=hs_encoder_hidden_states,
                attention_mask=hs_encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=hs_cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hs_hidden_states = nn.functional.dropout(hs_hidden_states, p=self.dropout, training=self.training)
            hs_hidden_states = residual + hs_hidden_states
            hs_hidden_states = self.ca_cross_attn_layer_norm(hs_hidden_states)
            hs_present_key_value = hs_present_key_value + hs_cross_attn_present_key_value

        ######### Information Fusion and FFN Block ########
        fused_hidden_states = self.fusion_layer(kg_hidden_states, hs_hidden_states, up_hidden_states)
        
        ffn_hidden_states = self.activation_fn(self.fc1(fused_hidden_states))
        ffn_hidden_states = nn.functional.dropout(ffn_hidden_states, p=self.activation_dropout, training=self.training)
        ffn_hidden_states = self.fc2(ffn_hidden_states)
        ffn_hidden_states = nn.functional.dropout(ffn_hidden_states, p=self.dropout, training=self.training)
        final_hidden_states = fused_hidden_states + ffn_hidden_states
        final_hidden_states = self.final_layer_norm(final_hidden_states)
        
        return final_hidden_states


class Planner(nn.Module):
    """
    Plan Decoder.
    """
    def __init__(self, args, embed_tokens=None):
        super(Planner, self).__init__()
        self.embed_dim = args.embed_dim
        self.vocab_size = args.vocab_size
        self.decoder_layers = args.decoder_layers
        self.dropout = args.dropout
        self.layerdrop = args.decoder_layerdrop
        self.padding_idx = args.pad_token_id
        self.max_position_embeddings = args.max_position_embeddings
        self.embed_scale = math.sqrt(args.embed_dim) if args.scale_embedding else 1.0
        self.init_std = args.init_std
        self.output_attentions = args.output_attentions
        self.output_hidden_states = args.output_hidden_states
        self.use_cache = args.use_cache

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(self.vocab_size, self.embed_dim, self.padding_idx)
            self._init_module_weights(self.embed_tokens)
        self.position_embed_layer = PlanPositionalEmbedding(self.max_position_embeddings, self.embed_dim)
        self.planner_layers = nn.ModuleList([DecoderLayer(args) for _ in range(self.decoder_layers)])
        self.plan_layer_norm = nn.LayerNorm(self.embed_dim)
        self.target_layer_norm = nn.LayerNorm(self.embed_dim)

        self._init_all_weights()
        self.gradient_checkpointing = False

    def _init_module_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.init_std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _init_all_weights(self):
        for layer in self.planner_layers:
            layer._init_all_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask

    def forward(
        self,
        input_ids=None,                     # (batch_size, seq_len)
        attention_mask=None,                # (batch_size, seq_len)
        target_ids=None,                    # (batch_size, target_seq_len)
        target_mask=None,                   # (batch_size, target_seq_len)
        kg_encoder_hidden_states=None,      # (batch_size, kg_seq_len, hidden_size)
        kg_encoder_attention_mask=None,     # (batch_size, kg_seq_len)
        up_encoder_hidden_states=None,      # (batch_size, up_seq_len, hidden_size)
        up_encoder_attention_mask=None,     # (batch_size, up_seq_len)
        hs_encoder_hidden_states=None,      # (batch_size, hs_seq_len, hidden_size)
        hs_encoder_attention_mask=None,     # (batch_size, hs_seq_len)
        head_mask=None,                     # (decoder_layers, decoder_attention_heads)
        cross_attn_head_mask=None,          # (decoder_layers, decoder_attention_heads)
        past_key_values=None,           
        inputs_embeds=None,                 # (batch_size, seq_len, hidden_size)
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=False
    ):
        self.device=input_ids.device

        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.use_cache

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            target_shape = target_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            target_ids = target_ids.view(-1, target_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
            target_embeds = self.embed_tokens(target_ids) * self.embed_scale
        
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )
        # expand encoder attention mask
        if kg_encoder_hidden_states is not None and kg_encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            kg_encoder_attention_mask = _expand_mask(kg_encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        # expand encoder attention mask
        if up_encoder_hidden_states is not None and up_encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            up_encoder_attention_mask = _expand_mask(up_encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        # expand encoder attention mask
        if hs_encoder_hidden_states is not None and hs_encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            hs_encoder_attention_mask = _expand_mask(hs_encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
        # expand weight encoder attention mask
        if inputs_embeds is not None and target_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            target_mask = _expand_mask_v2(target_mask, torch.int32, tgt_len=kg_encoder_attention_mask.size()[-1])

        # embedding layers
        inputs_position_embeds = self.position_embed_layer(input_shape, past_key_values_length)
        target_position_embeds = self.position_embed_layer(target_shape, past_key_values_length)
        hidden_states = inputs_embeds + inputs_position_embeds
        target_states = target_embeds + target_position_embeds
        hidden_states = self.plan_layer_norm(hidden_states)
        target_states = self.target_layer_norm(target_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        target_states = nn.functional.dropout(target_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and kg_encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (
                    len(self.planner_layers)
                ), f"The `{mask_name}` should be specified for {len(self.planner_layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.planner_layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    target_states=target_states,
                    target_mask=target_mask,
                    kg_encoder_hidden_states=kg_encoder_hidden_states,
                    kg_encoder_attention_mask=kg_encoder_attention_mask,
                    up_encoder_hidden_states=up_encoder_hidden_states,
                    up_encoder_attention_mask=up_encoder_attention_mask,
                    hs_encoder_hidden_states=hs_encoder_hidden_states,
                    hs_encoder_attention_mask=hs_encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions
            )
            hidden_states = layer_outputs
            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if kg_encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return DecoderOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def _expand_mask_v2(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    return expanded_mask

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)
import math
import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.file_utils import add_start_docstrings_to_model_forward
from .configuration_spt import SPTConfig

def repeat_kv(hidden_states, repeat_times):
    if repeat_times == 1:
        return hidden_states
    batch, n_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:,:,None,:,:].expand(batch, n_kv_heads, repeat_times, seq_len, head_dim)
    return hidden_states.reshape(batch, n_kv_heads*repeat_times, seq_len, head_dim)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.hidden_size // config.n_attn_heads
        kv_size = config.n_kv_heads * self.head_dim
        self.hidden_size = config.hidden_size
        self.n_attn_heads = config.n_attn_heads
        self.n_kv_heads = config.n_kv_heads

        self.q = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.v = nn.Linear(config.hidden_size, kv_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(config.max_len, config.max_len)).view(1, 1, config.max_len, config.max_len))

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        q = q.view(batch_size, seq_len, self.n_attn_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        k = repeat_kv(k, self.n_attn_heads//self.n_kv_heads)
        v = repeat_kv(v, self.n_attn_heads//self.n_kv_heads)

        attention = (q @ k.transpose(-2,-1)) * (1.0/math.sqrt(self.hidden_size))
        attention = attention.masked_fill(self.tril[:,:,:seq_len,:seq_len]==0, float('-inf'))
        probs = nn.functional.softmax(attention,dim=-1)
        y = probs@v
        y = y.transpose(1,2).contiguous().reshape(batch_size, seq_len, -1)

        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.GELU()

    def forward(self,x):
        up = self.up(x)
        gate = self.gate(x)
        return self.down(self.act_fn(up * gate))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        self.mlp = MLP(config)
        self.residual = config.residual
        self.norm = RMSNorm(config.hidden_size) if config.normalise else nn.Identity()

    def forward(self, x):
        if self.residual:
            x = x + self.attn(self.norm(x))
            x = x + self.mlp(self.norm(x))
        else:
            x = self.attn(self.norm(x))
            x = self.mlp(self.norm(x))
        return x

class SPTModel(PreTrainedModel):
    config_class = SPTConfig

    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.hidden_size) if config.normalise else nn.Identity()

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

class SPTForCausalLM(PreTrainedModel):
    config_class = SPTConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = SPTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        x = self.model(input_ids)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            hidden_states=x,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    @staticmethod
    def _reorder_cache(past, beam_idx):
        return past

# Register the custom model
AutoModelForCausalLM.register(SPTConfig, SPTForCausalLM)
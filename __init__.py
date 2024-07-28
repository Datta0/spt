from transformers import AutoConfig, AutoModelForCausalLM
from .configuration_spt import SPTConfig
from .modeling_spt import SPTForCausalLM

AutoConfig.register("spt", SPTConfig)
AutoModelForCausalLM.register(SPTConfig, SPTForCausalLM)

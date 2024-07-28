from transformers import PretrainedConfig, AutoConfig

class SPTConfig(PretrainedConfig):
    model_type = "spt"

    def __init__(
        self,
        vocab_size=97,
        hidden_size=512,
        n_layers=12,
        n_attn_heads=16,
        n_kv_heads=16,
        intermediate_size=2048,
        max_len=2048,
        residual=True,
        normalise=True,
        pad_token_id=95,
        bos_token_id=95,
        eos_token_id=95,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_attn_heads = n_attn_heads
        self.n_kv_heads = n_kv_heads
        self.intermediate_size = intermediate_size
        self.max_len = max_len
        self.residual = residual
        self.normalise = normalise
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )

AutoConfig.register("spt", SPTConfig)

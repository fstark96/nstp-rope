"""NSTP v2 wrapper — set dynamo disable BEFORE torch import."""
import os
os.environ['PYTORCH_DYNAMO_DISABLE'] = '1'
os.environ['TORCH_DYNAMO_DISABLE'] = '1'

import sys
# Remove any existing profile override
if 'profile' in sys.modules:
    del sys.modules['profile']

import torch
torch._dynamo.disable()
torch._dynamo.config.suppress_errors = True

from transformers import PreTrainedModel, PretrainedConfig
sys.path.insert(0, 'C:/Users/user/AppData/Local/Temp/nstp-v2')
from nstp_v2 import NSTPV2


class NSTPConfig(PretrainedConfig):
    model_type = "nstp"
    def __init__(self, vocab_size=50257, d_model=320, num_layers=3,
                 num_heads=4, hsa_dim=2048, num_experts=4, top_k=2,
                 d_ff=768, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hsa_dim = hsa_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_ff = d_ff
        self.dropout = dropout


class NSTPForCausalLM(PreTrainedModel):
    config_class = NSTPConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.nstp = NSTPV2(
            config.vocab_size, config.d_model, config.num_layers,
            config.num_heads, config.hsa_dim, config.num_experts,
            config.top_k, config.d_ff, config.dropout
        )
        self.lm_head = self.nstp.head
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        logits, _ = self.nstp(input_ids)
        return {'logits': logits}


if __name__ == "__main__":
    config = NSTPConfig()
    model = NSTPForCausalLM(config)
    ckpt = torch.load('C:/Users/user/AppData/Local/Temp/nstp-v2/models_scaled/finetune_best.pt', 
                       map_location='cpu', weights_only=True)
    model.nstp.load_state_dict(ckpt['model'], strict=False)
    model = model.cuda()
    x = torch.randint(0, 50257, (2, 128)).cuda()
    with torch.no_grad():
        out = model(x)
    print(f"Wrapper OK: {out['logits'].shape}")

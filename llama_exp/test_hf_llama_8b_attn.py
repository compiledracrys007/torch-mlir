import torch
from torch_mlir import fx
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_mlir.compiler_utils import run_pipeline_with_repro_report, lower_mlir_module, OutputType
import os
from compiler_tools import cpu_runner as ct

# -----------------------------------------------------
# 1. Load Model: LLaMA-3 8B (official Meta model)
# -----------------------------------------------------
MODEL = "meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, token=os.getenv("HF_TOKEN", None))

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",                     # or "auto" if you have a GPU
    trust_remote_code=True,
    token=os.getenv("HF_TOKEN", None)
)

# ----------------------------------------------------------------
# 🔥 Force attention scale = 1.0 for all layers
# ----------------------------------------------------------------
for layer in model.model.layers:
    if hasattr(layer.self_attn, "scaling"):
        layer.self_attn.scaling = 1.0
    if hasattr(layer.self_attn, "softmax_scale"):
        layer.self_attn.softmax_scale = 1.0

# Choose which layer to extract
LAYER_INDEX = 0   # first layer

attention_layer = model.model.layers[LAYER_INDEX].self_attn


# -----------------------------------------------------
# 2. Wrapper to simplify the forward() for export
# -----------------------------------------------------
class Llama3AttentionForExport(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, hidden_states):
        B, S, D = hidden_states.shape
        head_dim = self.m.head_dim
        
        cos = torch.ones(B, S, head_dim, dtype=hidden_states.dtype)
        sin = torch.zeros(B, S, head_dim, dtype=hidden_states.dtype)

        # HF LlamaAttention expects this
        pos_embeddings = (cos, sin)
        out = self.m(
               hidden_states=hidden_states,
               position_embeddings=pos_embeddings,
               attention_mask=None,
               position_ids=None,
               past_key_value=None,
               use_cache=False,
               cos=cos,    # required by eager_attention_forward
               sin=sin,    # required by eager_attention_forward
            )
        return out[0]

wrapped = Llama3AttentionForExport(attention_layer)

# -----------------------------------------------------
# 3. Dummy inputs (batch=1, seq=8)
# -----------------------------------------------------
hidden_dim = model.config.hidden_size
seq_len = 16
batch = 1

hidden_states = torch.randn(batch, seq_len, hidden_dim, dtype=torch.float16)

out = ct.compile_and_run(wrapped, (hidden_states))
print(out)

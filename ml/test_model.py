"""Quick sanity check for the Pebble model."""
import torch
from pebble.config import PebbleConfig
from pebble.model import PebbleLMHeadModel

print("=" * 60)
print("  PEBBLE — Model Verification")
print("=" * 60)

# 1. Create config and model
config = PebbleConfig()
model = PebbleLMHeadModel(config)

param_count = model.count_parameters()
print(f"  Parameters:  {param_count / 1e6:.1f}M")
print(f"  d_model:     {config.d_model}")
print(f"  n_layers:    {config.n_layers}")
print(f"  d_inner:     {config.d_inner}")
print(f"  d_state:     {config.d_state}")
print(f"  dt_rank:     {config.dt_rank}")
print(f"  vocab_size:  {config.vocab_size}")

# 2. Forward pass test
x = torch.randint(0, 1000, (2, 128))
result = model(input_ids=x, labels=x)
loss = result["loss"].item()
logits_shape = result["logits"].shape

print(f"\n  Forward pass: OK")
print(f"  Loss:         {loss:.4f}")
print(f"  Logits shape: {logits_shape}")

# 3. Generation cache test
cache = model.init_cache()
single_token = torch.randint(0, 1000, (1, 1))
result_cached = model(input_ids=single_token, cache=cache)
print(f"  Cached step:  OK (shape: {result_cached['logits'].shape})")

# 4. Estimate from config
estimate = config.num_params_estimate
print(f"\n  Config estimate: {estimate / 1e6:.1f}M")
print(f"  Actual count:    {param_count / 1e6:.1f}M")

print("\n" + "=" * 60)
print("  All checks passed.")
print("=" * 60)

"""Convert Wan-AI T5 checkpoint (HF-format keys) to musubi-tuner format."""
import sys
import torch


def convert_key(key: str) -> str | None:
    """Map HF T5 encoder key to musubi-tuner T5Encoder key."""
    # Skip non-encoder keys
    if key in ("spiece_model", "scaled_fp8"):
        return None

    # encoder.block.N.layer.0.SelfAttention.X -> blocks.N.attn.X
    # encoder.block.N.layer.0.layer_norm -> blocks.N.norm1
    # encoder.block.N.layer.1.DenseReluDense.wi_0 -> blocks.N.ffn.gate.0
    # encoder.block.N.layer.1.DenseReluDense.wi_1 -> blocks.N.ffn.fc1
    # encoder.block.N.layer.1.DenseReluDense.wo -> blocks.N.ffn.fc2
    # encoder.block.N.layer.1.layer_norm -> blocks.N.norm2
    # encoder.final_layer_norm -> norm
    # shared -> token_embedding
    # block.N.layer.0.SelfAttention.relative_attention_bias -> blocks.N.pos_embedding.embedding

    # Strip encoder. prefix if present
    k = key.replace("encoder.", "")

    if k == "shared.weight":
        return "token_embedding.weight"
    if k == "final_layer_norm.weight":
        return "norm.weight"

    if not k.startswith("block."):
        return None

    parts = k.split(".")
    block_idx = parts[1]
    layer_idx = parts[3]  # 0 = attention, 1 = ffn

    if layer_idx == "0":
        # Attention layer
        sublayer = parts[4]  # SelfAttention or layer_norm
        if sublayer == "layer_norm":
            return f"blocks.{block_idx}.norm1.{parts[5]}"
        elif sublayer == "SelfAttention":
            attn_part = parts[5]
            suffix = parts[6] if len(parts) > 6 else ""
            if attn_part == "relative_attention_bias":
                return f"blocks.{block_idx}.pos_embedding.embedding.{suffix}"
            # q, k, v, o mapping
            attn_map = {"q": "q", "k": "k", "v": "v", "o": "o"}
            if attn_part in attn_map:
                return f"blocks.{block_idx}.attn.{attn_map[attn_part]}.{suffix}"
            # scale_weight keys (for fp8)
            if attn_part.endswith("scale_weight"):
                return None  # Skip fp8 scale weights
            return None
    elif layer_idx == "1":
        sublayer = parts[4]
        if sublayer == "layer_norm":
            return f"blocks.{block_idx}.norm2.{parts[5]}"
        elif sublayer == "DenseReluDense":
            ffn_part = parts[5]
            suffix = parts[6] if len(parts) > 6 else ""
            ffn_map = {
                "wi_0": "ffn.gate.0",  # gated activation input
                "wi_1": "ffn.fc1",     # linear input
                "wo": "ffn.fc2",       # output
            }
            if ffn_part in ffn_map:
                return f"blocks.{block_idx}.{ffn_map[ffn_part]}.{suffix}"
            return None

    return None


def convert(input_path: str, output_path: str):
    print(f"Loading {input_path}...")
    sd = torch.load(input_path, map_location="cpu", weights_only=True)

    print(f"Original keys: {len(sd)}")
    # Show a few original keys
    for k in list(sd.keys())[:5]:
        print(f"  {k}")

    new_sd = {}
    skipped = []
    for key, value in sd.items():
        # Skip scale_weight keys (fp8 artifacts)
        if "scale_weight" in key:
            skipped.append(key)
            continue

        new_key = convert_key(key)
        if new_key is None:
            skipped.append(key)
            continue
        new_sd[new_key] = value

    print(f"\nConverted keys: {len(new_sd)}")
    for k in list(new_sd.keys())[:5]:
        print(f"  {k}")
    print(f"Skipped: {len(skipped)} keys")

    print(f"\nSaving to {output_path}...")
    torch.save(new_sd, output_path)
    print("Done!")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_t5_to_musubi.py <input.pth> <output.pth>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])

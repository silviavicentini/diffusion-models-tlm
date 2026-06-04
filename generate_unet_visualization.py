# %% [markdown]
# # Generate U-Net Architecture Visualization (HTML)
#
# This script generates an HTML file that shows the effect of U-Net
# on an input image

# %%
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diffusers import UNet2DModel
from torch.utils.data import Dataset
import base64
from io import BytesIO
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model setup ---
IMG_SIZE = 16
IMG_CHANNELS = 3
T_STEPS = 1000

betas = torch.linspace(1e-4, 0.02, T_STEPS).to(device)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)


def forward_sample(x0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    ab = alpha_bar[t].view(-1, 1, 1, 1)
    return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise, noise


def make_unet():
    return UNet2DModel(
        sample_size=IMG_SIZE,
        in_channels=IMG_CHANNELS,
        out_channels=IMG_CHANNELS,
        layers_per_block=2,
        block_out_channels=(64, 128, 256),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    )


model = make_unet().to(device)

# Try to load trained weights
CKPT_PATH = "diffusion_model_ema.pth"
if os.path.exists(CKPT_PATH):
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    print(f"Loaded trained model from {CKPT_PATH}")
else:
    # Try EMA weights
    CKPT_PATH2 = "diffusion_model_ema_weights.pth"
    if os.path.exists(CKPT_PATH2):
        model.load_state_dict(torch.load(CKPT_PATH2, map_location=device))
        print(f"Loaded EMA model from {CKPT_PATH2}")
    else:
        print("No checkpoint found — using random weights (structure demo only)")

model.eval()

# --- Load a sample image ---
SPRITES_PATH = "./data/sprites_1788_16x16.npy"
if os.path.exists(SPRITES_PATH):
    data = np.load(SPRITES_PATH)
    data = data[:len(data) // 50]
    images = torch.from_numpy(data).float() / 255.0
    images = (images - 0.5) * 2
    images = images.permute(0, 3, 1, 2)
    test_img = images[42].unsqueeze(0).to(device)
else:
    test_img = torch.randn(1, 3, 16, 16, device=device)

# --- Hook all sub-components ---
features = {}


def make_hook(name):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            features[name] = output[0].detach().cpu()
        else:
            features[name] = output.detach().cpu()
    return hook_fn


hooks = []

# Encoder L0
hooks.append(model.down_blocks[0].resnets[0].register_forward_hook(make_hook("enc0_res0")))
hooks.append(model.down_blocks[0].resnets[1].register_forward_hook(make_hook("enc0_res1")))
hooks.append(model.down_blocks[0].downsamplers[0].register_forward_hook(make_hook("enc0_down")))

# Encoder L1
hooks.append(model.down_blocks[1].resnets[0].register_forward_hook(make_hook("enc1_res0")))
hooks.append(model.down_blocks[1].attentions[0].register_forward_hook(make_hook("enc1_attn0")))
hooks.append(model.down_blocks[1].resnets[1].register_forward_hook(make_hook("enc1_res1")))
hooks.append(model.down_blocks[1].attentions[1].register_forward_hook(make_hook("enc1_attn1")))
hooks.append(model.down_blocks[1].downsamplers[0].register_forward_hook(make_hook("enc1_down")))

# Bottleneck
hooks.append(model.mid_block.resnets[0].register_forward_hook(make_hook("mid_res0")))
hooks.append(model.mid_block.attentions[0].register_forward_hook(make_hook("mid_attn")))
hooks.append(model.mid_block.resnets[1].register_forward_hook(make_hook("mid_res1")))

# Decoder L0
hooks.append(model.up_blocks[0].resnets[0].register_forward_hook(make_hook("dec0_res0")))
hooks.append(model.up_blocks[0].attentions[0].register_forward_hook(make_hook("dec0_attn0")))
hooks.append(model.up_blocks[0].resnets[1].register_forward_hook(make_hook("dec0_res1")))
hooks.append(model.up_blocks[0].attentions[1].register_forward_hook(make_hook("dec0_attn1")))
hooks.append(model.up_blocks[0].upsamplers[0].register_forward_hook(make_hook("dec0_up")))

# Decoder L1
hooks.append(model.up_blocks[1].resnets[0].register_forward_hook(make_hook("dec1_res0")))
hooks.append(model.up_blocks[1].attentions[0].register_forward_hook(make_hook("dec1_attn0")))
hooks.append(model.up_blocks[1].resnets[1].register_forward_hook(make_hook("dec1_res1")))
hooks.append(model.up_blocks[1].attentions[1].register_forward_hook(make_hook("dec1_attn1")))
hooks.append(model.up_blocks[1].upsamplers[0].register_forward_hook(make_hook("dec1_up")))

# Decoder L2
hooks.append(model.up_blocks[2].resnets[0].register_forward_hook(make_hook("dec2_res0")))
hooks.append(model.up_blocks[2].resnets[1].register_forward_hook(make_hook("dec2_res1")))

# Output head
hooks.append(model.conv_norm_out.register_forward_hook(make_hook("out_norm")))
hooks.append(model.conv_act.register_forward_hook(make_hook("out_act")))
hooks.append(model.conv_out.register_forward_hook(make_hook("out_conv")))

# --- Forward pass ---
t_test = torch.tensor([50], device=device)
x_noisy, _ = forward_sample(test_img, t_test)

with torch.no_grad():
    output = model(x_noisy, t_test).sample

for h in hooks:
    h.remove()

# Also store input and output as special entries
features["_input"] = x_noisy.cpu()
features["_output"] = output.cpu()


# %%
# --- Generate feature map images as base64 PNGs ---
def feat_to_base64(feat_tensor, is_rgb=False):
    """Convert a feature tensor to a base64-encoded PNG."""
    fig, ax = plt.subplots(1, 1, figsize=(1.2, 1.2), dpi=80)
    if is_rgb:
        img = feat_tensor[0].clamp(-1, 1) * 0.5 + 0.5
        ax.imshow(img.permute(1, 2, 0).numpy().clip(0, 1), interpolation="nearest")
    else:
        if feat_tensor.dim() == 4:
            img = feat_tensor[0].mean(0)
        else:
            img = feat_tensor.mean(0)
        ax.imshow(img.numpy(), cmap="inferno", interpolation="nearest")
    ax.axis("off")
    plt.tight_layout(pad=0)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# Generate all images
feat_images = {}
for key, tensor in features.items():
    is_rgb = key in ("_input", "_output", "out_conv")
    feat_images[key] = feat_to_base64(tensor, is_rgb=is_rgb)


# %%
# --- Generate HTML ---
def img_tag(key, size=60):
    """Create an img tag from a feature key."""
    return f'<img src="data:image/png;base64,{feat_images[key]}" width="{size}" height="{size}" style="border:1px solid #444; border-radius:3px;">'


def block_html(label, key, color="#2a2a3a"):
    """Create a styled block with label and feature image."""
    return f'''<div class="block" style="background:{color};">
        <div class="block-label">{label}</div>
        {img_tag(key)}
        <div class="block-info">{features[key].shape[-2]}×{features[key].shape[-1]}, {features[key].shape[-3]}ch</div>
    </div>'''


def arrow_down():
    return '<div class="arrow">▼</div>'


def arrow_right():
    return '<span class="arrow-h">→</span>'


html = '''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>U-Net Architecture — Feature Map Visualization</title>
<style>
body {
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: #1a1a2e;
    color: #e0e0e0;
    margin: 0;
    padding: 20px;
}
h1 { text-align: center; color: #64ffda; margin-bottom: 5px; }
h2 { color: #bb86fc; margin: 20px 0 10px 0; }
.subtitle { text-align: center; color: #888; margin-bottom: 30px; }
.container { max-width: 1400px; margin: 0 auto; }

.section {
    background: #16213e;
    border-radius: 12px;
    padding: 20px;
    margin: 15px 0;
    border: 1px solid #333;
}
.section-title {
    font-size: 14px;
    font-weight: bold;
    color: #64ffda;
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.flow-row {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 6px;
    justify-content: center;
}

.block {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    padding: 8px;
    border-radius: 8px;
    border: 1px solid #555;
    min-width: 80px;
}
.block-label {
    font-size: 10px;
    font-weight: bold;
    color: #fff;
    margin-bottom: 4px;
    text-align: center;
}
.block-info {
    font-size: 9px;
    color: #aaa;
    margin-top: 3px;
}

.arrow {
    text-align: center;
    color: #64ffda;
    font-size: 18px;
    margin: 4px 0;
}
.arrow-h {
    color: #64ffda;
    font-size: 20px;
    margin: 0 2px;
}

.skip-label {
    font-size: 10px;
    color: #ff9800;
    font-style: italic;
    text-align: center;
    margin: 4px 0;
}

.resblock { background: #1b3a4b; }
.attnblock { background: #3a1b4b; }
.downblock { background: #4b3a1b; }
.upblock { background: #1b4b3a; }
.normblock { background: #3a3a1b; }
.inputblock { background: #0a3d0a; }

.legend {
    display: flex;
    gap: 15px;
    justify-content: center;
    margin: 15px 0;
    flex-wrap: wrap;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
}
.legend-swatch {
    width: 14px;
    height: 14px;
    border-radius: 3px;
    border: 1px solid #555;
}
</style>
</head>
<body>
<div class="container">
<h1>U-Net Architecture — Live Feature Maps</h1>
<p class="subtitle">Input sprite with light noise (t=50) passed through the trained model. Each image shows mean activation across channels.</p>

<div class="legend">
    <div class="legend-item"><div class="legend-swatch" style="background:#0a3d0a;"></div> Input/Output</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#1b3a4b;"></div> ResBlock</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#3a1b4b;"></div> Attention</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#4b3a1b;"></div> Downsample</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#1b4b3a;"></div> Upsample</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#3a3a1b;"></div> Norm/Act</div>
</div>
'''

# --- Input ---
html += '''
<div class="section">
<div class="section-title">Input</div>
<div class="flow-row">
'''
html += f'''<div class="block inputblock">
    <div class="block-label">x_t (t=50)</div>
    {img_tag("_input", 80)}
    <div class="block-info">16×16, 3ch</div>
</div>'''
html += '</div></div>'

# --- Encoder Level 0 ---
html += '''
<div class="section">
<div class="section-title">Encoder Level 0 — DownBlock2D (16×16 → 8×8, 64ch)</div>
<div class="flow-row">
'''
html += f'''<div class="block resblock"><div class="block-label">ResBlock</div>{img_tag("enc0_res0")}<div class="block-info">16×16, 64ch</div></div>'''
html += arrow_right()
html += f'''<div class="block resblock"><div class="block-label">ResBlock</div>{img_tag("enc0_res1")}<div class="block-info">16×16, 64ch</div></div>'''
html += arrow_right()
html += f'''<div class="block downblock"><div class="block-label">↓2 Conv</div>{img_tag("enc0_down")}<div class="block-info">8×8, 64ch</div></div>'''
html += '</div>'
html += '<div class="skip-label">── skip connection saved (16×16, 64ch) ──</div>'
html += '</div>'

# --- Encoder Level 1 ---
html += '''
<div class="section">
<div class="section-title">Encoder Level 1 — AttnDownBlock2D (8×8 → 4×4, 128ch)</div>
<div class="flow-row">
'''
html += f'''<div class="block resblock"><div class="block-label">ResBlock</div>{img_tag("enc1_res0")}<div class="block-info">8×8, 128ch</div></div>'''
html += arrow_right()
html += f'''<div class="block attnblock"><div class="block-label">Attention</div>{img_tag("enc1_attn0")}<div class="block-info">8×8, 128ch</div></div>'''
html += arrow_right()
html += f'''<div class="block resblock"><div class="block-label">ResBlock</div>{img_tag("enc1_res1")}<div class="block-info">8×8, 128ch</div></div>'''
html += arrow_right()
html += f'''<div class="block attnblock"><div class="block-label">Attention</div>{img_tag("enc1_attn1")}<div class="block-info">8×8, 128ch</div></div>'''
html += arrow_right()
html += f'''<div class="block downblock"><div class="block-label">↓2 Conv</div>{img_tag("enc1_down")}<div class="block-info">4×4, 128ch</div></div>'''
html += '</div>'
html += '<div class="skip-label">── skip connection saved (8×8, 128ch) ──</div>'
html += '</div>'

# --- Bottleneck ---
html += '''
<div class="section">
<div class="section-title">Bottleneck (4×4, 256ch)</div>
<div class="flow-row">
'''
html += f'''<div class="block resblock"><div class="block-label">ResBlock</div>{img_tag("mid_res0")}<div class="block-info">4×4, 256ch</div></div>'''
html += arrow_right()
html += f'''<div class="block attnblock"><div class="block-label">Self-Attn</div>{img_tag("mid_attn")}<div class="block-info">4×4, 256ch</div></div>'''
html += arrow_right()
html += f'''<div class="block resblock"><div class="block-label">ResBlock</div>{img_tag("mid_res1")}<div class="block-info">4×4, 256ch</div></div>'''
html += '</div></div>'

# --- Decoder Level 0 ---
html += '''
<div class="section">
<div class="section-title">Decoder Level 0 — AttnUpBlock2D (4×4 → 8×8, 256→128ch)</div>
<div class="skip-label">── skip connection concatenated (from encoder L1) ──</div>
<div class="flow-row">
'''
html += f'''<div class="block resblock"><div class="block-label">ResBlock</div>{img_tag("dec0_res0")}<div class="block-info">4×4, 256ch</div></div>'''
html += arrow_right()
html += f'''<div class="block attnblock"><div class="block-label">Attention</div>{img_tag("dec0_attn0")}<div class="block-info">4×4, 256ch</div></div>'''
html += arrow_right()
html += f'''<div class="block resblock"><div class="block-label">ResBlock</div>{img_tag("dec0_res1")}<div class="block-info">4×4, 256ch</div></div>'''
html += arrow_right()
html += f'''<div class="block attnblock"><div class="block-label">Attention</div>{img_tag("dec0_attn1")}<div class="block-info">4×4, 256ch</div></div>'''
html += arrow_right()
html += f'''<div class="block upblock"><div class="block-label">↑2 Upsample</div>{img_tag("dec0_up")}<div class="block-info">8×8, 256ch</div></div>'''
html += '</div></div>'

# --- Decoder Level 1 ---
html += '''
<div class="section">
<div class="section-title">Decoder Level 1 — AttnUpBlock2D (8×8 → 16×16, 128ch)</div>
<div class="skip-label">── skip connection concatenated (from encoder L0) ──</div>
<div class="flow-row">
'''
html += f'''<div class="block resblock"><div class="block-label">ResBlock</div>{img_tag("dec1_res0")}<div class="block-info">8×8, 128ch</div></div>'''
html += arrow_right()
html += f'''<div class="block attnblock"><div class="block-label">Attention</div>{img_tag("dec1_attn0")}<div class="block-info">8×8, 128ch</div></div>'''
html += arrow_right()
html += f'''<div class="block resblock"><div class="block-label">ResBlock</div>{img_tag("dec1_res1")}<div class="block-info">8×8, 128ch</div></div>'''
html += arrow_right()
html += f'''<div class="block attnblock"><div class="block-label">Attention</div>{img_tag("dec1_attn1")}<div class="block-info">8×8, 128ch</div></div>'''
html += arrow_right()
html += f'''<div class="block upblock"><div class="block-label">↑2 Upsample</div>{img_tag("dec1_up")}<div class="block-info">16×16, 128ch</div></div>'''
html += '</div></div>'

# --- Decoder Level 2 ---
html += '''
<div class="section">
<div class="section-title">Decoder Level 2 — UpBlock2D (16×16, 64ch) + Output Head</div>
<div class="flow-row">
'''
html += f'''<div class="block resblock"><div class="block-label">ResBlock</div>{img_tag("dec2_res0")}<div class="block-info">16×16, 64ch</div></div>'''
html += arrow_right()
html += f'''<div class="block resblock"><div class="block-label">ResBlock</div>{img_tag("dec2_res1")}<div class="block-info">16×16, 64ch</div></div>'''
html += arrow_right()
html += f'''<div class="block normblock"><div class="block-label">GroupNorm</div>{img_tag("out_norm")}<div class="block-info">16×16, 64ch</div></div>'''
html += arrow_right()
html += f'''<div class="block normblock"><div class="block-label">SiLU</div>{img_tag("out_act")}<div class="block-info">16×16, 64ch</div></div>'''
html += arrow_right()
html += f'''<div class="block inputblock"><div class="block-label">Conv 1×1</div>{img_tag("out_conv")}<div class="block-info">16×16, 3ch</div></div>'''
html += '</div></div>'

# --- Output ---
html += '''
<div class="section">
<div class="section-title">Output — Predicted Noise ε̂</div>
<div class="flow-row">
'''
html += f'''<div class="block inputblock">
    <div class="block-label">ε̂ (predicted noise)</div>
    {img_tag("_output", 80)}
    <div class="block-info">16×16, 3ch</div>
</div>'''
html += '</div></div>'

html += '''
</div>
</body>
</html>
'''

# --- Write HTML ---
viz_dir  = ".data/viz"
os.makedirs(viz_dir, exist_ok=True)
output_path = f"{viz_dir}/unet_processing.html"
with open(output_path, "w") as f:
    f.write(html)

print(f"Generated: {output_path}")
print(f"Open in browser to view the architecture with live feature maps.")

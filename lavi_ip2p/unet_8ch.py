"""
UNet 8-channel modification for LaVi-IP2P

Safely converts LaVi-Bridge UNet from 4-channel to 8-channel input
by expanding conv_in with zero-initialized extra channels (InstructPix2Pix style).
"""

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from typing import Optional


def convert_unet_to_8ch(
    unet: UNet2DConditionModel,
    init_scale: float = 1e-5,
) -> UNet2DConditionModel:
    """
    Convert UNet from 4-channel to 8-channel input.
    
    Strategy (InstructPix2Pix style):
    1. Copy first 4 channels from pretrained conv_in weights
    2. Initialize extra 4 channels with small random values (scaled by init_scale)
    3. This ensures the UNet starts close to the original behavior
    
    Args:
        unet: Pretrained 4-channel UNet
        init_scale: Scale for initializing extra channels (default: 1e-5 for stability)
    
    Returns:
        Modified 8-channel UNet
    """
    
    if unet.config.in_channels == 8:
        print("⚠️  UNet already has 8 input channels, skipping conversion")
        return unet
    
    if unet.config.in_channels != 4:
        raise ValueError(f"Expected 4-channel UNet, got {unet.config.in_channels}")
    
    print("\n" + "="*60)
    print("Converting UNet to 8-channel input (IP2P style)")
    print("="*60)
    
    old_conv = unet.conv_in
    print(f"Original conv_in: {old_conv}")
    print(f"  in_channels: {old_conv.in_channels}")
    print(f"  out_channels: {old_conv.out_channels}")
    print(f"  kernel_size: {old_conv.kernel_size}")
    print(f"  weight shape: {old_conv.weight.shape}")
    
    # Create new 8-channel conv
    new_conv = nn.Conv2d(
        in_channels=8,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=(old_conv.bias is not None),
    )
    
    # Initialize weights
    with torch.no_grad():
        # Copy first 4 channels from pretrained weights
        new_conv.weight[:, :4, :, :].copy_(old_conv.weight)
        
        # Initialize extra 4 channels with small random values
        # This ensures the UNet starts close to original behavior
        nn.init.normal_(new_conv.weight[:, 4:, :, :], mean=0.0, std=init_scale)
        
        # Copy bias if present
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    
    # Replace conv_in
    unet.conv_in = new_conv
    
    # Update config
    unet.register_to_config(in_channels=8)
    
    print(f"\n✓ New conv_in: {new_conv}")
    print(f"  weight shape: {new_conv.weight.shape}")
    print(f"  Channels 0-3: copied from pretrained")
    print(f"  Channels 4-7: initialized with std={init_scale}")
    print("="*60 + "\n")
    
    return unet


def load_lavi_unet_8ch(
    pretrained_path: str,
    lora_path: Optional[str] = None,
    lora_rank: int = 32,
    init_scale: float = 1e-5,
    device: str = "cuda",
) -> UNet2DConditionModel:
    """
    Load LaVi-Bridge UNet with LoRA and convert to 8-channel.
    
    Args:
        pretrained_path: Path to pretrained SD UNet (e.g., "CompVis/stable-diffusion-v1-4")
        lora_path: Path to LaVi-Bridge LoRA weights (lora_vis.pt)
        lora_rank: LoRA rank (default: 32)
        init_scale: Scale for extra channels
        device: Device to load on
    
    Returns:
        8-channel UNet with LaVi-Bridge LoRA applied
    """
    
    print("\n" + "="*60)
    print("Loading LaVi-Bridge UNet (8-channel)")
    print("="*60)
    
    # Load base UNet
    print(f"\n[1/3] Loading base UNet from {pretrained_path}...")
    
    # Determine subfolder based on filesystem check
    from pathlib import Path
    subfolder = "unet" if Path(pretrained_path).is_dir() and (Path(pretrained_path) / "unet").exists() else None
    
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_path,
        subfolder=subfolder,
    )
    print(f"✓ Base UNet loaded (subfolder: {subfolder})")
    
    # Apply LoRA if provided
    if lora_path is not None:
        print(f"\n[2/3] Applying LaVi-Bridge LoRA from {lora_path}...")
        
        from lavi_bridge.modules.lora import monkeypatch_or_replace_lora_extended
        
        VIS_REPLACE_MODULES = {"ResnetBlock2D", "CrossAttention", "Attention", "GEGLU"}
        
        monkeypatch_or_replace_lora_extended(
            unet,
            torch.load(lora_path, map_location='cpu'),
            r=lora_rank,
            target_replace_module=VIS_REPLACE_MODULES,
        )
        print(f"✓ LoRA applied (rank={lora_rank})")
    else:
        print(f"\n[2/3] No LoRA path provided, using base UNet")
    
    # Convert to 8-channel
    print(f"\n[3/3] Converting to 8-channel input...")
    unet = convert_unet_to_8ch(unet, init_scale=init_scale)
    
    # Move to device
    unet = unet.to(device)
    print(f"✓ Moved to {device}")
    print("="*60 + "\n")
    
    return unet


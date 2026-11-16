"""
LLaVA Token Adapter for LaVi-IP2P

Adapts cached LLaVA hidden states (4096-dim) to LaVi-Bridge UNet cross-attention (768-dim).
Uses LaVi-Bridge's TextAdapter with optional preprocessing (RMSNorm, scaling).
"""

import torch
import torch.nn as nn
from typing import Optional

from lavi_bridge.modules.adapters import TextAdapter


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class LLaVAAdapter(nn.Module):
    """
    Adapter: LLaVA hidden (4096) → LaVi-Bridge UNet conditioning (768).
    
    FROZEN by default (use pretrained LaVi-Bridge TextAdapter as-is).
    """
    
    def __init__(
        self,
        adapter_path: str,
        use_rms_norm: bool = False,
        learnable_scale: bool = False,
        init_scale: float = 1.0,
        freeze_adapter: bool = True,
    ):
        super().__init__()
        
        self.use_rms_norm = use_rms_norm
        self.learnable_scale = learnable_scale
        
        # Load pretrained TextAdapter
        print(f"\nLoading TextAdapter from {adapter_path}...")
        self.adapter = TextAdapter.from_pretrained(adapter_path)
        print(f"✓ TextAdapter loaded (4096→768)")
        
        # Freeze by default
        if freeze_adapter:
            for param in self.adapter.parameters():
                param.requires_grad = False
            print("  ✓ FROZEN (using pretrained weights)")
        else:
            print("  ⚠️  TRAINABLE")
        
        # Optional: RMSNorm
        if use_rms_norm:
            self.rms_norm = RMSNorm(4096, eps=1e-6)
            print("✓ RMSNorm enabled")
        
        # Optional: Learnable scale
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(init_scale))
            print(f"✓ Learnable scale (init={init_scale})")
        else:
            self.register_buffer('scale', torch.tensor(init_scale))
    
    def forward(self, llava_hidden: torch.Tensor) -> torch.Tensor:
        """LLaVA (B,L,4096) → UNet conditioning (B,L,768)"""
        x = llava_hidden
        
        if self.use_rms_norm:
            x = self.rms_norm(x)
        
        x = self.adapter(x).sample
        
        if self.learnable_scale:
            x = x * self.scale
        
        return x
    
    def get_stats(self):
        """Get adapter statistics for logging"""
        stats = {}
        if self.learnable_scale:
            stats['scale'] = self.scale.item()
        return stats


def create_llava_adapter(
    adapter_path: str,
    use_rms_norm: bool = False,
    learnable_scale: bool = False,
    init_scale: float = 1.0,
    freeze_adapter: bool = True,
) -> LLaVAAdapter:
    """
    Factory: Create LLaVA adapter.
    
    DEFAULT (simple mode):
      - Freeze TextAdapter (use LaVi-Bridge pretrained)
      - No RMSNorm, no scale
      - Train UNet only
    """
    
    print("\n" + "="*60)
    print("Creating LLaVA Token Adapter")
    print("="*60)
    print(f"Mode: {'SIMPLE (frozen)' if freeze_adapter else 'ADVANCED (trainable)'}")
    
    adapter = LLaVAAdapter(
        adapter_path=adapter_path,
        use_rms_norm=use_rms_norm,
        learnable_scale=learnable_scale,
        init_scale=init_scale,
        freeze_adapter=freeze_adapter,
    )
    
    total = sum(p.numel() for p in adapter.parameters())
    trainable = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    
    print(f"\nParameters: {total:,} total, {trainable:,} trainable")
    print("="*60 + "\n")
    
    return adapter


import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

# Register additional activation
ACT2FN["softsign"] = nn.Softsign

from utils.config import DictConfig, update_config
from utils.datasets import DATASET_CHANNELS
from models.model_output import ModelOutput
from models.model_utils import (
    create_context_mask,
    SmoothAndNoise,
    NeuralMLP,
    NeuralAttention,
    NeuralFactorsProjection,
)

DEFAULT_CONFIG = "configs/models/encoder/ndt.yaml"

@dataclass
class NDTOutput(ModelOutput):
    loss: torch.FloatTensor = None
    n_examples: torch.LongTensor = None
    preds: torch.FloatTensor = None
    targets: torch.FloatTensor = None
    mask: Optional[torch.LongTensor] = None


class ReadIn(nn.Module):
    """
    Dataset-specific read-in module.
    
    Applies: LayerNorm -> Linear -> LayerNorm to each patch,
    with dataset-specific input dimensions based on `channel_dict`.
    """
    def __init__(
            self, channel_dict: Dict[str, int], hidden_size: int, patch_size: int = 5
        ):
        super().__init__()
        self.hidden_size = hidden_size
        self.act = ACT2FN["softsign"]

        # Build dataset-specific embedding layers
        self.embed_dict = nn.ModuleDict({
            str(dataset_idx): self.patch_embedder(channel_count, hidden_size, patch_size)
            for dataset_idx, channel_count in enumerate(channel_dict.values())
        })

    @staticmethod
    def patch_embedder(n_channels: int, hidden_size: int, patch_size: int) -> nn.Sequential:
        """Creates LayerNorm -> Linear -> LayerNorm projection to hidden size."""
        patch_dim = n_channels * patch_size
        return nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, hidden_size),
            nn.LayerNorm(hidden_size)
        )

    def forward(self, x: torch.Tensor, dataset_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): shape [B, T_patch, C x patch_size].
            dataset_idx (Tensor): dataset index tensor (same for all in batch).

        Returns:
            Tensor: embedded input (from patched neural data).
        """
        dataset_idx = torch.unique(dataset_idx).item()
        return self.embed_dict[str(dataset_idx)](x)


class ReadOut(nn.Module):
    """
    Dataset-specific read-out module.

    If `n_outputs` is provided, outputs a fixed dimension for all datasets.
    Otherwise, projects back to the original (patched) input dimension for each dataset.
    """
    def __init__(
        self,
        channel_dict: Dict[str, int], 
        hidden_size: int, 
        n_outputs: Optional[int] = None,
        patch_size: int = 5
    ):
        super().__init__()

        self.readout_dict = nn.ModuleDict({
            str(dataset_idx): (
                nn.Linear(hidden_size, n_outputs) if n_outputs is not None
                else self.patch_projector(channel_count, hidden_size, patch_size)
            )
            for dataset_idx, channel_count in enumerate(channel_dict.values())
        })

    @staticmethod
    def patch_projector(n_channels: int, hidden_size: int, patch_size: int) -> nn.Sequential:
        """Creates LayerNorm -> Linear -> LayerNorm projection to patch dimension."""
        patch_dim = n_channels * patch_size
        return nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, patch_dim),
            nn.LayerNorm(patch_dim)
        )

    def forward(self, x: torch.Tensor, dataset_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input features of shape [B, T_patch, D_hidden].
            dataset_idx (Tensor): Dataset index tensor (same for all in batch).

        Returns:
            Tensor: projected output (back to the patched neural data).
        """
        dataset_idx = torch.unique(dataset_idx).item()
        return self.readout_dict[str(dataset_idx)](x)

    
class DecoderWrapper(nn.Module):
    """
    Wrapper for dataset-specific readout followed by optional downstream layers.
    """
    def __init__(
        self,
        read_out: nn.Module,
        downstream_layers: Optional[Sequence[nn.Module]] = None
    ):
        super().__init__()
        self.read_out = read_out
        self.downstream = (
            nn.Sequential(*downstream_layers) if downstream_layers else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        dataset_idx: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if dataset_idx is None:
            x = self.read_out(x)
        else:
            x = self.read_out(x, dataset_idx)
        return self.downstream(x)


class NeuralEmbeddingLayer(nn.Module):
    """
    Patch and embed neural data, and apply optional masking.
    """
    def __init__(
        self,
        hidden_size: int,
        channel_dict: Dict[str, int],
        config: DictConfig
    ):
        super().__init__()

        # Patching
        self.patch_size = config.patch_size

        # Masking
        self.mask_active = config.masker.get("active", False)
        self.mask_ratio = config.masker.ratio
        self.expand_prob = config.masker.expand_prob
        self.max_timespan = config.masker.max_timespan

        # Layers
        self.read_in = ReadIn(channel_dict, hidden_size, self.patch_size)
        self.dropout = nn.Dropout(config.dropout)
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_size))

    def pad_for_patching(
        self,
        spikes: torch.Tensor,
        spikes_mask: torch.Tensor,
        spikes_timestamp: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Pad input tensors along the time dimension so that T is divisible by `patch_size`.
        """
        B, T, C = spikes.shape
        pad_len = (self.patch_size - (T % self.patch_size)) % self.patch_size

        if pad_len > 0:
            # Pad spikes
            pad_spikes = torch.full(
                (B, pad_len, C), -100, device=spikes.device, dtype=spikes.dtype
            )
            spikes = torch.cat([spikes, pad_spikes], dim=1)

            # Pad mask
            if spikes_mask is not None:
                pad_mask = torch.zeros(
                    B, pad_len, device=spikes_mask.device, dtype=spikes_mask.dtype
                )
                spikes_mask = torch.cat([spikes_mask, pad_mask], dim=1)

            # Pad timestamps
            if spikes_timestamp is not None:
                pad_timestamp = torch.zeros(
                    B, pad_len, device=spikes_timestamp.device, dtype=spikes_timestamp.dtype
                )
                spikes_timestamp = torch.cat([spikes_timestamp, pad_timestamp], dim=1)

            # Pad targets
            if targets is not None:
                pad_targets = torch.full(
                    (B, pad_len, C), -100, device=targets.device, dtype=targets.dtype
                )
                targets = torch.cat([targets, pad_targets], dim=1)

        return spikes, spikes_mask, spikes_timestamp, targets

    def patch_data(
        self, 
        spikes: torch.Tensor,
        spikes_mask: torch.Tensor,
        spikes_timestamp: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Reshape from (B, T, C) to (B, T_patch, patch_size * C).
        """
        B, T, C = spikes.shape

        if T % self.patch_size != 0:
            raise ValueError(f"T={T} is not divisible by patch_size={self.patch_size}")

        num_patches = T // self.patch_size
        spikes = spikes.reshape(B, num_patches, self.patch_size * C)

        if targets is not None:
            targets = targets.reshape(B, num_patches, self.patch_size * C)

        spikes_mask = spikes_mask.reshape(B, num_patches, self.patch_size).any(dim=2).long()
        
        spikes_timestamp = spikes_timestamp[:, 0::self.patch_size]

        return spikes, spikes_mask, spikes_timestamp, targets
    
    def masker(
        self, 
        batch_size: int, 
        num_patches: int, 
        device: torch.device,
    ) -> torch.Tensor:
        """
        Generate a boolean mask for patched sequences, optionally expanding to mask consecutive patches.
        """
        # Compute patch span length
        if torch.bernoulli(torch.tensor(self.expand_prob).float()):
            span = torch.randint(1, self.max_timespan + 1, (1,)).item() 
        else:
            span = 1

        # Adjust mask ratio based on span length
        mask_ratio_per_span = self.mask_ratio / span
        num_masked_spans = int(mask_ratio_per_span * num_patches)

        mask = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=device)

        # Generate mask for each trial in the batch
        for b in range(batch_size):
            # Random starting indices for each span
            starts = torch.randperm(num_patches - span + 1, device=device)[:num_masked_spans]
            for s in starts:
                mask[b, s:s + span] = True

        return mask

    def apply_mask(
        self, x: torch.Tensor, force_mask: bool =True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Mask patched data.
        """
        batch_size, num_patches, _ = x.shape

        if self.mask_active and force_mask:
            targets_mask = self.masker(batch_size, num_patches, x.device)
            mask_tokens = self.mask_token.expand(batch_size, num_patches, -1)
            x = torch.where(targets_mask.unsqueeze(-1), mask_tokens, x)
        else:
            targets_mask = None

        return x, targets_mask

    def forward(
        self,
        spikes: torch.FloatTensor,
        spikes_mask: torch.LongTensor,
        spikes_timestamp: torch.LongTensor,
        dataset_name: Optional[torch.Tensor] = None,
        targets: Optional[torch.FloatTensor] = None,
        force_mask: bool = True,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, Optional[torch.LongTensor], Optional[torch.FloatTensor]]:
        spikes, spikes_mask, spikes_timestamp, targets = self.pad_for_patching(
            spikes, spikes_mask, spikes_timestamp, targets
        )

        spikes, spikes_mask, spikes_timestamp, targets = self.patch_data(
            spikes, spikes_mask, spikes_timestamp, targets
        )

        x = self.read_in(spikes, dataset_name)

        x, targets_mask = self.apply_mask(x, force_mask)

        x = self.dropout(x)

        return x, spikes_mask, spikes_timestamp, targets_mask, targets
    
    
class NeuralEncoderLayer(nn.Module):
    """
    Transformer encoder layer, consisting of:
    - LayerNorm + multi-head attention
    - LayerNorm + MLP
    - Optional RoPE (Rotary Positional Embeddings)
    - Optional Fixup initialization
    """

    def __init__(self, layer_idx: int, max_F: int, config: DictConfig):
        super().__init__()

        self.idx = layer_idx
        self.use_rope = config.use_rope

        # Attention block
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attn = NeuralAttention(
            idx=self.idx,
            hidden_size=config.hidden_size,
            n_heads=config.n_heads,
            use_bias=config.attention_bias,
            dropout=config.dropout,
            use_rope=config.use_rope,
            base=config.rope_theta,
            max_F=max_F
        )

        # Feed-forward block
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.mlp = NeuralMLP(
            hidden_size=config.hidden_size,
            inter_size=config.inter_size,
            act=config.act,
            use_bias=config.mlp_bias,
            dropout=config.dropout
        )

        # Optional Fixup initialization
        if config.fixup_init:
            self.fixup_initialization(config.n_layers)

    def forward(
        self,
        x: torch.FloatTensor,
        attn_mask: torch.LongTensor,
        timestamp: Optional[torch.LongTensor] = None
    ) -> torch.FloatTensor:
        # Multi-head attention
        x = x + self.attn(
            self.ln1(x), attn_mask, timestamp if self.use_rope else None
        )
        # Feed-forward MLP
        x = x + self.mlp(self.ln2(x))
        return x

    def fixup_initialization(self, n_layers: int) -> None:
        """
        Apply Fixup initialization to specific parameters.
        """
        temp_state_dict = {}
        scale = 0.67 * (n_layers ** (-0.25))

        for name, param in self.named_parameters():
            if name.endswith("_proj.weight"):
                temp_state_dict[name] = scale * param
            elif name.endswith("value.weight"):
                temp_state_dict[name] = scale * (param * (2 ** 0.5))

        # Preserve parameters not affected by Fixup
        for name, value in self.state_dict().items():
            temp_state_dict.setdefault(name, value)

        self.load_state_dict(temp_state_dict)  


class NeuralEncoder(nn.Module):
    """Transformer encoder for neural activity.

    Applies:
        - Data augmentation (smoothing + noise injection)
        - Patching and embedding neural sequences
        - Stacked Transformer layers
    """
    def __init__(
        self, 
        channel_dict: Dict[int, int],
        config: DictConfig,
    ):
        super().__init__() 

        # Transformer config
        self.hidden_size = config.transformer.hidden_size
        self.n_layers = config.transformer.n_layers

        # Data augmentation
        self.smooth_and_noise = SmoothAndNoise(config.smooth_and_noise)

        # Patching and embedding
        self.embedder = NeuralEmbeddingLayer(
            hidden_size=self.hidden_size,
            channel_dict=channel_dict,
            config=config.embedder
        )

        # Context mask for causal or acausal attention
        context_mask = create_context_mask(
            config.context.forward, config.context.backward, config.embedder.max_F
        )
        self.register_buffer("context_mask", context_mask, persistent=False)

        # Transformer layers
        self.layers = nn.ModuleList([
            NeuralEncoderLayer(
                layer_idx=layer_idx, max_F=config.embedder.max_F, config=config.transformer
            )
            for layer_idx in range(self.n_layers)
        ])

        # Output
        self.out_norm = nn.LayerNorm(self.hidden_size)
        self.out_proj = NeuralFactorsProjection(self.hidden_size, config.factors)

    def forward(
        self,
        spikes: torch.FloatTensor,             # (B, T, N)
        spikes_mask: torch.LongTensor,         # (B, T)
        spikes_timestamp: torch.LongTensor,    # (B, T)
        dataset_name: Optional[torch.Tensor] = None,
        targets: Optional[torch.FloatTensor] = None,
        force_mask: Optional[bool] = True,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor, Optional[torch.FloatTensor]]:
    
        spikes = self.smooth_and_noise(spikes)

        x, spikes_mask, spikes_timestamp, targets_mask, targets = self.embedder(
            spikes, spikes_mask, spikes_timestamp, dataset_name, targets, force_mask
        )

        B, T, _ = x.size()

        # Create attention mask
        context_mask = self.context_mask[:T, :T].expand(B, T, T).to(x.device)
        self_mask = torch.eye(T, device=x.device, dtype=torch.int64).expand(B, T, T)
        # `spikes_mask` is used to prevent attention to padded patches
        attn_mask = self_mask | (context_mask & spikes_mask.unsqueeze(1))

        # Transformer layers
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, timestamp=spikes_timestamp)

        x = self.out_norm(x)

        x = self.out_proj(x)

        return x, spikes_mask, spikes_timestamp, targets_mask, targets


class NDT(nn.Module):
    """Neural Data Transformer.

    Supports:
        - SSL with masking.
        - CTC for supervised learning.
    """
    def __init__(
        self, 
        config: DictConfig,
        **kwargs
    ):
        super().__init__()

        # Update default config with provided config 
        config = update_config(DEFAULT_CONFIG, config)
        self.config = config
        self.method = kwargs["method_name"]
        channel_dict = DATASET_CHANNELS[kwargs["features"]]

        # Load pretrained encoder config if provided
        encoder_pt_path = config["encoder"].pop("from_pt", None)
        if encoder_pt_path is not None:
            encoder_config = torch.load(os.path.join(encoder_pt_path, "encoder_config.pth"))
            config["encoder"] = update_config(config.encoder, encoder_config)
            print(f"Load pretrained NDT from {encoder_pt_path}")

        # Create encoder
        self.encoder = NeuralEncoder(channel_dict, config.encoder)

        # Load pretrained encoder weights if provided
        if encoder_pt_path is not None:
            self.encoder.load_state_dict(
                torch.load(os.path.join(encoder_pt_path, "encoder.bin")), strict=False
            )

        # Determine output size based on training method
        n_outputs = kwargs["vocab_size"] if self.method in ["ctc"] else None

        # Create decoder
        read_out = ReadOut(channel_dict, self.encoder.out_proj.out_size, n_outputs)
        
        downstream_layers = []

        if self.method in ["ssl"]:
            if kwargs["loss"] == "poisson_nll" and not kwargs["log_input"]:
                downstream_layers.append(nn.ReLU())
        elif self.method in ["ctc", "endtoend"]:
            downstream_layers.append(nn.LogSoftmax(dim=-1))
        else:
            raise ValueError(f"Method {self.method} not supported.")

        self.decoder = DecoderWrapper(read_out, downstream_layers)

        # Load pretrained decoder weights if provided
        if (encoder_pt_path is not None) and (self.method not in ["ctc"]):
            self.decoder.load_state_dict(
                torch.load(
                    os.path.join(encoder_pt_path, "decoder.bin")), strict=False
            )

        # Loss function
        if self.method in ["ssl"]:
            if kwargs["loss"] == "poisson_nll":
                self.loss_fn = nn.PoissonNLLLoss(
                    reduction="none", log_input=kwargs["log_input"]
                )
            elif kwargs["loss"] == "mse":
                self.loss_fn = nn.MSELoss(reduction="none")
            else:
                raise ValueError(f"Loss {kwargs['loss']} not implemented for {self.method}")
        elif self.method in ["ctc"]:
            self.loss_fn = nn.CTCLoss(
                reduction="none", blank=kwargs["blank_id"], zero_infinity=kwargs["zero_infinity"]
            )

    def forward(
        self,
        spikes: torch.FloatTensor,
        spikes_mask: torch.LongTensor,
        spikes_timestamp: torch.LongTensor,
        targets: Optional[torch.FloatTensor] = None,
        targets_lengths: Optional[torch.LongTensor] = None,
        dataset_name: Optional[torch.Tensor] = None,
    ) -> NDTOutput:

        B, _, N = spikes.shape

        # For SSL, use spikes as targets
        if self.method in ["ssl"]:
            force_mask = True 
            targets = spikes.clone()

        if self.method in ["ctc"]: 
            force_mask = self.training
            targets_labels = targets.clone()
            targets = None

        x, spikes_mask, spikes_timestamp, targets_mask, targets = self.encoder(
            spikes, spikes_mask, spikes_timestamp, dataset_name, targets, force_mask
        )

        preds = self.decoder(x, dataset_name)

        if self.method in ["ssl"]:
            # Re-compute `targets_mask` for loss calculation
            padding_mask = (targets != -100)
            targets_mask = (targets_mask & spikes_mask).unsqueeze(2)
            targets_mask = (targets_mask & padding_mask).reshape(B, -1, N)
            preds = preds.reshape(B, -1, N)
            targets = targets.reshape(B, -1, N)

            loss = (self.loss_fn(preds, targets) * targets_mask).sum()
            n_examples = targets_mask.sum()

        elif self.method in ["ctc"]:
            input_lengths = spikes_mask.sum(dim=1).to(torch.long)

            loss = self.loss_fn(
                log_probs=preds.transpose(0, 1),
                targets=targets_labels,
                input_lengths=input_lengths,
                target_lengths=targets_lengths
            ).sum()

            n_examples = targets_lengths

        return NDTOutput(
            loss=loss,
            n_examples=n_examples,
            preds=preds,
            targets=targets,
            mask=targets_mask,
        )

    def save_checkpoint(self, save_dir: str) -> None:
        """Save encoder and decoder states along with encoder config."""
        torch.save(self.encoder.state_dict(), os.path.join(save_dir, "encoder.bin"))
        torch.save(dict(self.config.encoder), os.path.join(save_dir, "encoder_config.pth"))
        torch.save(self.decoder.state_dict(), os.path.join(save_dir, "decoder.bin"))

    def load_checkpoint(self, load_dir: str, strict=False) -> None:
        """Load encoder and decoder states from checkpoint directory."""
        self.encoder.load_state_dict(
            torch.load(os.path.join(load_dir, "encoder.bin")), strict=strict
        )
        self.decoder.load_state_dict(
            torch.load(os.path.join(load_dir, "decoder.bin")), strict=strict
        )
        
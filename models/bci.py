import os
import math
import warnings
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft import LoraConfig, get_peft_model
from transformers import Qwen2AudioForConditionalGeneration
from transformers.activations import ACT2FN

from models.ndt import NDT
from models.model_output import ModelOutput
from utils.config import DictConfig, update_config

ACT2FN["softsign"] = nn.Softsign
HF_TOKEN = os.environ.get("HF_TOKEN", None)

DEFAULT_CONFIG = "configs/models/projector/bci-audio.yaml"

NAME2MODEL = {"NDT": NDT, "BCI_AudioLLM": None}

@dataclass
class BCIOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    n_examples: Optional[torch.LongTensor] = None
    preds: Optional[torch.FloatTensor] = None
    targets: Optional[torch.FloatTensor] = None
    mask: Optional[torch.LongTensor] = None


class ModalityAligner(nn.Module):
    def __init__(self, embed_dim, proj_dim=512):
        super().__init__()
        self.spike_proj = nn.Linear(embed_dim, proj_dim)
        self.text_proj = nn.Linear(embed_dim, proj_dim)
    
    def forward(self, spikes_embeds, text_embeds):
        """
        spikes_embeds: [B, T, embed_dim]
        text_embeds:   [B, L, embed_dim]
        """
        # Reduce sequence to one token per modality
        spike_token = spikes_embeds.mean(dim=1)     # [B, embed_dim]
        text_token = text_embeds.mean(dim=1)        # [B, embed_dim]
        
        # Project into shared latent space
        spike_token = self.spike_proj(spike_token)  # [B, proj_dim]
        text_token = self.text_proj(text_token)     # [B, proj_dim]
        
        # Normalize for contrastive similarity
        spike_token = F.normalize(spike_token, dim=-1)
        text_token = F.normalize(text_token, dim=-1)
        
        return spike_token, text_token

class ContrastiveLoss(nn.Module):
    def __init__(self, init_temp: float = 0.1, max_scale: float = 100.0, reduction: str = "none"):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / init_temp)))
        self.max_scale = max_scale
        self.reduction = reduction

    def forward(self, spike_token: torch.Tensor, text_token: torch.Tensor) -> torch.Tensor:
        # Cosine similarity
        logits = spike_token @ text_token.T
        logit_scale = self.logit_scale.exp().clamp(max=self.max_scale)
        logits = logits * logit_scale

        labels = torch.arange(logits.size(0), device=logits.device)

        # Symmetric InfoNCE
        loss_s2t = F.cross_entropy(logits, labels, reduction="none")  
        loss_t2s = F.cross_entropy(logits.T, labels, reduction="none") 
        loss = (loss_s2t + loss_t2s) / 2  

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # [B]


class BCI_AudioLLM(nn.Module):
    """End-to-end BCI with a Qwen backend.
    """
    def __init__(
        self,
        config: DictConfig,
        llm_path: str,
        lora: Optional[Dict] = None,
        freeze_llm: bool = False,
        **kwargs,
    ):
        super().__init__()
        
        # Update config
        config = update_config(DEFAULT_CONFIG, config)
        self.config = config
        pt_path = dict(config).pop("from_pt", None)
        
        # Load pre-trained audio LLM (causal)
        self.llm = self.load_llm(llm_path, pt_path, lora, freeze_llm)
        self.llm_config = self.llm.config

        # Build neural encoder
        neural_encoder = self.load_neural_encoder(
            config, pt_path, kwargs
        )
        self.neural_encoder = neural_encoder

        # Build projector
        self.project_to_audio = kwargs.get("project_to_audio", False)
        self.projector = self.build_projector(config, pt_path)

        # (Optional) Build modality aligner for contrastive loss
        if kwargs.get("use_contrastive", False):
            self.modality_aligner = ModalityAligner(
                embed_dim=config.projector.embed_size * 2
            )
            self.contrastive_loss_fn = ContrastiveLoss(reduction="sum")
        else:
            self.modality_aligner = None
            self.contrastive_loss_fn = None
        
        self.loss_fn = nn.CrossEntropyLoss(reduction="sum")

    def load_llm(self, llm_path, pt_path, lora, freeze_llm):
        llm = Qwen2AudioForConditionalGeneration.from_pretrained(
            pt_path or llm_path,
            use_auth_token=HF_TOKEN
        )
        if lora and pt_path is None:
            peft = DictConfig(lora)
            peft_config = LoraConfig(
                inference_mode=False,
                r=peft.r,
                lora_alpha=peft.alpha,
                lora_dropout=peft.dropout,
                target_modules=peft.target_modules,
                modules_to_save=peft.modules_to_save,
            )
            llm = get_peft_model(llm, peft_config)

        if freeze_llm:
            for p in llm.parameters():
                p.requires_grad = False
        return llm
    
    def disable_masking(self, encoder):
        encoder.encoder.embedder.mask_active = False
        encoder.encoder.embedder.mask_ratio = 0.0
        encoder.encoder.smooth_and_noise.noise = False
        encoder.encoder.smooth_and_noise.smooth = 2.0
        return encoder
    
    def load_safetensors(self, encoder, checkpoint_dir, device="cpu", strict=True):
        from safetensors.torch import load_file as load_safetensors
        model_path = f"{checkpoint_dir}/model.safetensors"
        state_dict = load_safetensors(model_path, device=device)
        extra_keys = encoder.load_state_dict(state_dict, strict=strict)
        return encoder
    
    def load_neural_encoder(self, config, pt_path, kwargs):
        encoder = None

        name = config["encoder"]["model_class"]
        model_cls = NAME2MODEL[name]
        name = name.lower()
        encoder_pt_path = (
            pt_path 
            or kwargs.get(f"load_{name}_from_pt", None) 
            or config["encoder"]["encoder"].get("from_pt", None)
        )
        if encoder_pt_path:
            config["encoder"]["encoder"]["from_pt"] = encoder_pt_path
            config["encoder"]["decoder"]["from_pt"] = encoder_pt_path
            print(f"Load {name.upper()} from: {encoder_pt_path}")
        encoder = model_cls(config.encoder, **kwargs)

        if encoder_pt_path:
            encoder = self.load_safetensors(encoder, encoder_pt_path, strict=False)

        encoder = self.disable_masking(encoder)

        return encoder
    
    def get_hidden_size(self):
        if isinstance(self.neural_encoder, NDT):
            return self.neural_encoder.encoder.hidden_size
        else:
            raise ValueError(
                f"Unsupported neural encoder type: {type(self.neural_encoder)}."
            )
        
    def build_projector(self, config, pt_path):
        if pt_path:
            proj_cfg = torch.load(os.path.join(pt_path, "projector_config.pth"))
            config["projector"] = update_config(config.projector, proj_cfg)

        hidden_size = self.get_hidden_size()

        if self.project_to_audio:
            self.allm_hidden_size = self.llm_config.audio_config.d_model
        else:
            self.allm_hidden_size = self.llm_config.text_config.hidden_size
        
        if config.projector.inter_size is not None:
            projector = nn.Sequential(
                nn.Linear(
                    hidden_size * config.projector.stacking,
                    config.projector.inter_size,
                    bias=config.projector.bias
                ),
                ACT2FN[config.projector.act],
                nn.Linear(
                    config.projector.inter_size,
                    self.allm_hidden_size,
                    bias=config.projector.bias
                )
            )
        else:
            projector = nn.Linear(
                hidden_size * config.projector.stacking,
                self.allm_hidden_size,
                bias=config.projector.bias
            )

        if pt_path:
            projector.load_state_dict(
                torch.load(os.path.join(pt_path, "projector.bin"))
            )
        return projector
    
    def prepare_neural_embeds(
        self,
        spikes: torch.FloatTensor,
        spikes_mask: torch.LongTensor,
        spikes_timestamp: torch.LongTensor,
        dataset_name: Optional[torch.Tensor] = None, 
        day_idx: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        
        if isinstance(self.neural_encoder, NDT):
            spikes_embeds, spikes_mask, _, _, _ = self.neural_encoder.encoder(
                spikes, spikes_mask, spikes_timestamp, dataset_name, force_mask=False
            )
        else:
            raise ValueError(f"Unsupported neural encoder type: {type(self.neural_encoder)}.")
        
        return spikes_embeds, spikes_mask
    
    def merge_embeds(
        self,
        text_embeds: torch.FloatTensor,
        spikes_embeds: torch.FloatTensor,
        spikes_mask: torch.LongTensor,
        attention_mask: torch.LongTensor,
        input_split: torch.LongTensor,
        targets: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
        
        inputs_embeds = torch.stack([
            torch.cat((t[:d], s, t[d:]), dim=0)
            for t, s, d in zip(text_embeds, spikes_embeds, input_split)
        ], dim=0)

        attention_mask = torch.stack([
            torch.cat((a[:d], s, a[d:]), dim=0)
            for a, s, d in zip(attention_mask, spikes_mask, input_split)
        ], dim=0)

        if targets is not None:
            targets = torch.stack([
                torch.cat((t[:d], torch.ones_like(s)*(-100), t[d:]), dim=0)
                for t, s, d in zip(targets, spikes_mask, input_split)
            ], dim=0)

        return inputs_embeds, attention_mask, targets
          
    def prepare_embeds(
        self,
        input_ids: torch.LongTensor,
        spikes: torch.FloatTensor,
        spikes_mask: torch.LongTensor,
        spikes_timestamp: torch.FloatTensor,
        dataset_name: Optional[torch.Tensor] = None,
        day_idx: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
        # Text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)

        # Neural embeddings
        spikes_embeds, spikes_mask = self.prepare_neural_embeds(
            spikes, spikes_mask, spikes_timestamp, dataset_name, day_idx
        )

        if self.project_to_audio:
            spikes_embeds = self.projector(spikes_embeds)
            spikes_embeds = self.llm.multi_modal_projector(spikes_embeds)
        else:
            spikes_embeds = self.projector(spikes_embeds)

        # (Optional) Align modality and compute contrastive loss
        if self.modality_aligner:
            spike_token, text_token = self.modality_aligner(spikes_embeds, text_embeds)
            contrastive_loss = self.contrastive_loss_fn(spike_token, text_token)
        else:
            spike_token, text_token = None, None
            contrastive_loss = None

        return spikes_embeds, text_embeds, spikes_mask, contrastive_loss

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        input_split: torch.LongTensor,
        spikes: torch.FloatTensor,
        spikes_mask: torch.LongTensor,
        spikes_timestamp: torch.LongTensor,
        spikes_lengths: torch.LongTensor,
        targets: Optional[torch.LongTensor] = None,
        dataset_name: Optional[torch.Tensor] = None,
        day_idx: Optional[torch.LongTensor] = None,
    ) -> BCIOutput:
        
        spikes_embeds, text_embeds, spikes_mask, contrastive_loss = self.prepare_embeds(
            input_ids, spikes, spikes_mask, spikes_timestamp, dataset_name, day_idx,
        )
        # Merge embeddings
        inputs_embeds, attention_mask, targets = self.merge_embeds(
            text_embeds, spikes_embeds, spikes_mask, attention_mask, input_split, targets
        )

        logits = self.llm(
            inputs_embeds=inputs_embeds, 
            attention_mask=attention_mask, 
            return_dict=True,
        ).logits

        loss, n_examples = None, None

        if targets is not None:
            shift_logits = logits[..., :-1, :].reshape(-1, self.llm_config.text_config.vocab_size)
            shift_targets = targets[..., 1:].contiguous().reshape(-1)
            loss = self.loss_fn(shift_logits.float(), shift_targets.long())
            n_examples = (shift_targets != -100).sum()

        if (contrastive_loss is not None) and (loss is not None):
            loss += contrastive_loss

        return BCIOutput(
            loss=loss,
            n_examples=n_examples,
            preds=logits,
            targets=targets,
        )

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        input_split: torch.LongTensor,
        spikes: torch.FloatTensor,
        spikes_mask: torch.LongTensor,
        spikes_timestamp: torch.LongTensor,
        spikes_lengths: torch.LongTensor,
        targets: Optional[torch.LongTensor] = None,
        dataset_name: Optional[torch.Tensor] = None,
        day_idx: Optional[torch.LongTensor] = None,
        inputs_embeds:      Optional[torch.FloatTensor] = None, 
        **gen_config:       DictConfig,
    ) -> List[torch.LongTensor]:  
        
        if inputs_embeds is None:
            spikes_embeds, text_embeds, spikes_mask, contrastive_loss = self.prepare_embeds(
                input_ids, spikes, spikes_mask, spikes_timestamp, dataset_name, day_idx,
            )
            # Merge embeddings
            inputs_embeds, attention_mask, targets = self.merge_embeds(
                text_embeds, spikes_embeds, spikes_mask, attention_mask, input_split, targets
            )
        # LLM built-in generation
        inputs_embeds = inputs_embeds.to(self.llm.dtype)

        return self.llm.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **gen_config
        )
    
    def save_checkpoint(self, save_dir):
        self.llm.save_pretrained(save_dir)
        self.neural_encoder.save_checkpoint(save_dir)
        torch.save(self.projector.state_dict(), os.path.join(save_dir, "projector.bin"))
        torch.save(dict(self.config.projector), os.path.join(save_dir, "projector_config.pth"))

    def load_checkpoint(self, load_dir):
        self.llm = Qwen2AudioForConditionalGeneration.from_pretrained(load_dir).to(self.llm.device)
        self.neural_encoder.load_checkpoint(load_dir)
        self.projector.load_state_dict(torch.load(os.path.join(load_dir, "projector.bin")))


NAME2MODEL.update(
    {"BCI_AudioLLM": BCI_AudioLLM}
)
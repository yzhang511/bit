import os 
import yaml
import json
import wandb
import inspect
from functools import partial
from tqdm import tqdm

import re

from typing import Optional, Union, Dict, List, Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from transformers import get_linear_schedule_with_warmup

import deepspeed
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from utils.config import DictConfig, update_config
from utils.datasets import (
    SpikingDataset, 
    SpikingDatasetForDecoding, 
    pad_collate_fn,
    GroupBatchSampler,
)
from models.ndt import NDT
from models.bci import BCI_AudioLLM

from utils.eval_spike import plot_gt_pred, plot_neurons_r2

NAME2DATASET = {
    "ssl": SpikingDataset, 
    "sl": SpikingDatasetForDecoding, 
}

NAME2MODEL = {"NDT": NDT, "BCI_AudioLLM": BCI_AudioLLM}

from utils.config import default_trainer_config


class Trainer():
    def __init__(
        self,
        config:             DictConfig,
        model:              Optional[nn.Module] = None,
        dataset:            Optional[Union[str,Dict[str,List[Dict[str,Any]]]]] = None,
        metric_fns:         Optional[Dict[str,Callable]] = None,
        extra_model_kwargs: Optional[Dict[str,Any]] = None,
        ds_config:          Optional[Union[str,Dict[str,Any]]] = None,
        eval_mode:          Optional[bool] = False,
    ):  
        self.config = update_config(default_trainer_config(), config) 

        self.verbosity = config.verbosity

        self.set_seed()
        
        self.set_mixed_precision()

        self.init_accelerator(ds_config)

        self.init_wandb()

        self.print_v(yaml.dump(dict(self.config), allow_unicode=True, default_flow_style=False), verbosity=0) 

        self.eval_mode = eval_mode

        self.prepare_logging()

        self.set_model(model, extra_model_kwargs)

        self.get_model_inputs()

        self.set_dataset(dataset)

        self.build_dataloaders()        

        self.build_optimizer_and_scheduler()
        
        self.prepare_for_distributed_training()
        
        self.metric_kwargs = self.config.method.metric_kwargs
        self.metric_fns = metric_fns if metric_fns else {}

        self.start_epoch = 1
        self.print_v("Starting training from scratch ...", verbosity=1)

    def print_v(self, *args, verbosity=3):
        # Print messages with the appropriate level of verbosity
        if verbosity >= self.verbosity:
            if self.accelerator.is_main_process:
                self.accelerator.print(*args)

    def set_seed(self):
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(self.config.seed)

    def set_mixed_precision(self):
        self.mixed_precision = self.config.training.mixed_precision
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, None: torch.float32}[self.mixed_precision]

    def init_accelerator(self, ds_config=None):
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        if ds_config is not None:
            from accelerate import DeepSpeedPlugin
            ds_plugin = DeepSpeedPlugin(hf_ds_config=ds_config.get("deepspeed_config", {}))
            self.accelerator = Accelerator(
                kwargs_handlers=[kwargs],
                step_scheduler_with_optimizer=self.config.optimizer.scheduler in ["linear", "cosine"],
                split_batches=True,
                mixed_precision=self.config.training.mixed_precision,
                deepspeed_plugin=ds_plugin
            )
        else:
            self.accelerator = Accelerator(
                kwargs_handlers=[kwargs],
                step_scheduler_with_optimizer=self.config.optimizer.scheduler in ["linear", "cosine"], 
                split_batches=True,
                mixed_precision=self.config.training.mixed_precision
            )
        self.world_size = self.accelerator.state.num_processes
        local_rank = self.accelerator.state.local_process_index
        dsitributed = not self.accelerator.state.distributed_type.value == "NO"
        self.print_v(f"Distributed: {dsitributed}, Local Rank: {local_rank}, World Size: {self.world_size}, Device: {self.accelerator.device}")

    def init_wandb(self):
        if self.config.log_to_wandb:
            if self.accelerator.is_main_process:
                self.wandb_run = wandb.init(
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity, 
                    name=self.config.savestring,
                    config=self.config,
                )

    def prepare_logging(self):
        self.savestring = self.config.savestring
        self.checkpoint_dir = os.path.join(self.config.dirs.checkpoint_dir, self.savestring)
        if not self.eval_mode:
            if not os.path.exists(self.checkpoint_dir) and self.accelerator.is_main_process:
                os.makedirs(self.checkpoint_dir)


    def set_model(self, model, extra_model_kwargs=None):
        if extra_model_kwargs is None:
            extra_model_kwargs = {}
        
        if model is None:
            model_class = NAME2MODEL[self.config.model.model_class]
            self.model = model_class(
                self.config.model, 
                **self.config.method.model_kwargs, 
                **self.config.method.dataset_kwargs,
                **extra_model_kwargs
            )
        else:
            self.model = model

        self.model.to(self.device_type)
        self.print_v(self.model)

        self.count_parameters(self.model, exclude_layers=["decoder", "embed_dict"])

        self.print_v(f"Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}", verbosity=0)

    def get_model_inputs(self):
        model_to_inspect = self.model
        signature = inspect.signature(model_to_inspect.forward)
        self.model_inputs = list(signature.parameters.keys())
        
    def set_dataset(self, dataset):
        if dataset is None:
            raise Exception("No dataset provided")
        elif isinstance(dataset, str):
            raise Exception("Can't load dataset from provided path or name")
        else:
            self.dataset = dataset

    def build_dataloaders(self):
        self.print_v("Building dataloaders", verbosity=0)

        dataset_class = NAME2DATASET[self.config.data.dataset_class]

        self.train_dataset = dataset_class(
            self.dataset, self.config.data.train_name, length=self.config.data.train_len, **self.config.method.dataset_kwargs
        )
        self.val_dataset = dataset_class(
            self.dataset, self.config.data.val_name, length=self.config.data.val_len, **self.config.method.dataset_kwargs
        )

        train_sampler = GroupBatchSampler(
            self.train_dataset, 
            batch_size=self.config.training.train_batch_size,
            shuffle=True,
            drop_last=self.config.training.drop_last_train_dataloader,
            seed=self.config.seed
        )

        val_sampler = GroupBatchSampler(
            self.val_dataset, 
            batch_size=self.config.training.test_batch_size,
            shuffle=self.config.training.shuffle_test_dataloader,
            drop_last=self.config.training.drop_last_train_dataloader,
            seed=self.config.seed
        )

        self.train_dataloader = DataLoader(
            self.train_dataset, 
            batch_sampler=train_sampler, 
            collate_fn=partial(pad_collate_fn, model_inputs=self.model_inputs, **self.config.method.dataloader_kwargs), 
            pin_memory=True, 
        )

        self.val_dataloader = DataLoader(
            self.val_dataset, 
            batch_sampler=val_sampler, 
            collate_fn=partial(pad_collate_fn, model_inputs=self.model_inputs, **self.config.method.dataloader_kwargs), 
            pin_memory=True, 
        )

    def build_optimizer_and_scheduler(self):
        self.print_v("Building optimizers", verbosity=0)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.optimizer.lr, 
            weight_decay=self.config.optimizer.wd, 
            eps=self.config.optimizer.eps
        )
        if self.config.optimizer.scheduler == "linear":
            self.lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=round(
                    self.config.optimizer.warmup_pct * self.config.training.num_epochs * len(self.train_dataloader) // self.config.optimizer.gradient_accumulation_steps
                ),
                num_training_steps=self.config.training.num_epochs * len(self.train_dataloader) // self.config.optimizer.gradient_accumulation_steps,
            )
        elif self.config.optimizer.scheduler == "cosine":
            self.lr_scheduler = OneCycleLR(
                optimizer=self.optimizer,
                total_steps=self.config.training.num_epochs * len(self.train_dataloader) // self.config.optimizer.gradient_accumulation_steps,
                max_lr=self.config.optimizer.lr,
                pct_start=self.config.optimizer.warmup_pct,
                div_factor=self.config.optimizer.div_factor,
            )
        else:
            raise Exception(f"Scheduler '{self.config.optimizer.scheduler}' not implemented")

    def prepare_for_distributed_training(self):
        self.print_v("Preparing for distributed training", verbosity=0)
        self.model, self.train_dataloader, self.val_dataloader, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.train_dataloader, self.val_dataloader, self.optimizer, self.lr_scheduler
        )

    def train(self):
        config = self.config
        global_step = 1
        best_metric = - float("inf")
        best_loss = float("inf")
        train_loss = []
        train_n_examples = []
        train_metrics = {name: [] for name in self.metric_fns.keys()}

        for epoch in range(self.start_epoch, config.training.num_epochs+1):
            self.print_v(f"Epoch {epoch}", verbosity=1)
            self.model.train()

            for step, (model_inputs, unused_inputs) in enumerate(tqdm(self.train_dataloader)):

                if (global_step-1) % config.optimizer.gradient_accumulation_steps == 0:
                    with torch.autocast(device_type=self.device_type, dtype=self.dtype):
                        outputs = self.model(**model_inputs)
                        n_examples = outputs.n_examples
                        loss = outputs.loss / config.optimizer.gradient_accumulation_steps
                        
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    if config.optimizer.scheduler in ["linear", "cosine"]:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                else:
                    with self.accelerator.no_sync(self.model):
                        with torch.autocast(device_type=self.device_type, dtype=self.dtype):
                            outputs = self.model(**model_inputs)
                            n_examples = outputs.n_examples
                            loss = outputs.loss / config.optimizer.gradient_accumulation_steps
                            self.accelerator.backward(loss)

                train_loss.append(self.accelerator.gather(loss).sum().detach().item())
                train_n_examples.append(self.accelerator.gather(n_examples).sum().detach().item())

                global_step += 1    

            if config.training.eval_every and (epoch % config.training.eval_every == 0):

                self.print_v(f"Eval Epoch {epoch}", verbosity=1)

                test_avg_loss, test_avg_metrics, gt, preds = self.evaluate()
                train_avg_loss = sum(train_loss) / sum(train_n_examples) if sum(train_n_examples) > 0 else 0
            
                self.print_v(f"{train_avg_loss=}" + "\n" + f"{test_avg_loss=}" + "\n" + f"{test_avg_metrics=}", verbosity=1) 
                
                if self.config.method.model_kwargs.method_name in ["ssl"]:
                    val_metric = test_avg_metrics[self.metric_kwargs.metric_name]
                elif self.config.method.model_kwargs.method_name in ["ctc", "endtoend"]:
                    val_metric = 1 - test_avg_metrics[self.metric_kwargs.metric_name]

                if self.config.method.model_kwargs.method_name in ["ssl"]:
                    active_neurons = list(range(gt.shape[-1])) 
                    gt_pred_fig = self.plot_epoch(
                        gt=gt, preds=preds, epoch=epoch, active_neurons=active_neurons[:5],
                    )

                if config.log_to_wandb:
                    if self.accelerator.is_main_process:
                        wandb.log({
                            "epoch": epoch, "train_avg_loss": train_avg_loss, 
                            "test_avg_loss": test_avg_loss, **test_avg_metrics,
                        })
                        if self.config.method.model_kwargs.method_name in ["ssl"]:
                            wandb.log({
                                f"test_population_raster_plot": wandb.Image(gt_pred_fig["plot_gt_pred"]),
                                f"test_single_neuron_plot": wandb.Image(gt_pred_fig["plot_r2"])
                            })

                if val_metric > best_metric:

                    best_metric = val_metric

                    self.save_checkpoint(save_name="BEST", epoch=epoch)
                    self.save_accelerator_state_to_resume(save_name="BEST", epoch=epoch)

                    if self.config.method.model_kwargs.method_name in ["ssl"]:
                        active_neurons = list(range(gt.shape[-1])) 
                        gt_pred_fig = self.plot_epoch(
                            gt=gt, preds=preds, epoch=epoch, active_neurons=active_neurons[:5],
                        )
                        if config.log_to_wandb:
                            if self.accelerator.is_main_process:
                                wandb.log({
                                    "best_epoch": epoch, "best_val_metric": best_metric,
                                })   
                                if self.config.method.model_kwargs.method_name in ["ssl"]:
                                    wandb.log({
                                        f"best_population_raster_plot": wandb.Image(gt_pred_fig["plot_gt_pred"]),
                                        f"best_single_neuron_plot": wandb.Image(gt_pred_fig["plot_r2"])
                                    })   

                train_loss = []
                train_n_examples = []
                train_metrics = {name: [] for name in self.metric_fns.keys()}

                self.model.train()     

            if config.training.save_every and (epoch % config.training.save_every == 0):
                self.save_checkpoint(save_name="EPOCH", epoch=epoch)
                self.save_accelerator_state_to_resume(save_name="EPOCH", epoch=epoch)

        self.save_checkpoint(save_name="LAST", epoch=epoch)
        self.save_accelerator_state_to_resume(save_name="LAST", epoch=epoch)
        self.print_v("Training completed!", verbosity=1)

    def evaluate(
        self,
        additional_metric_fns: Optional[Dict[str,Callable]] = None,
        eval_train_set: Optional[bool] = False,
    ):
        metric_fns = dict(**self.metric_fns)
        metric_fns.update(additional_metric_fns if additional_metric_fns else {})
        
        test_loss = []
        test_n_examples = []
        test_metrics = {name: [] for name in metric_fns.keys()}
        
        self.model.eval()
        
        dataloader = self.val_dataloader if not eval_train_set else self.train_dataloader

        for test_step, (model_inputs, unused_inputs) in enumerate(tqdm(dataloader)):
            
            with torch.no_grad() as A, self.accelerator.no_sync(self.model) as B:
                model_inputs = {k: v.to(dtype=self.dtype) if torch.is_tensor(v) and v.dtype == torch.float32 else v 
                                for k, v in model_inputs.items()}
                outputs = self.model(**model_inputs)
                n_examples = outputs.n_examples
                loss = outputs.loss
                test_loss.append(self.accelerator.gather(loss).sum().detach().item())
                test_n_examples.append(self.accelerator.gather(n_examples).sum().detach().item())

                for name, fn in metric_fns.items():
                    metric_value = fn(self.model, model_inputs, unused_inputs, outputs.to_dict(), self.config, **self.metric_kwargs)
                    reduced_metric = self.accelerator.reduce(metric_value, reduction="mean")
                    test_metrics[name].append(reduced_metric.detach().item())

            if self.config.method.model_kwargs.method_name in ["ssl"]:
                gt = outputs.targets
                if self.config.method.model_kwargs.loss == "mse":
                    preds = outputs.preds
                else:
                    preds = torch.exp(outputs.preds)
                mask = (gt == self.config.method.dataloader_kwargs.pad_dict["spikes"][0]["value"])
                # Replace padded values with 0 for visualization
                gt = gt.masked_fill(mask, 0).float().detach().cpu().numpy()
                preds = preds.masked_fill(mask, 0).float().detach().cpu().numpy()
            else:
                gt = outputs.targets
                preds = outputs.preds
            
        test_avg_loss = sum(test_loss) / sum(test_n_examples) if sum(test_n_examples) > 0 else 0
        test_avg_metrics = {k: sum(v)/len(v) for k, v in test_metrics.items()}

        return test_avg_loss, test_avg_metrics, gt, preds   

    def plot_epoch(self, gt, preds, epoch, active_neurons=None):
        gt_pred_fig = plot_gt_pred(
            gt = gt.mean(0).T,
            pred = preds.mean(0).T,
            epoch = epoch,
        )
        r2_fig = plot_neurons_r2(
            gt = gt.mean(0),
            pred = preds.mean(0),
            neuron_idx=active_neurons,
            epoch = epoch
        )
        return {
            "plot_gt_pred": gt_pred_fig,
            "plot_r2": r2_fig
        }

    def count_parameters(self, model, exclude_layers=None):
        if exclude_layers is None:
            exclude_layers = ["read_in", "decoder"]
        
        total_params = 0
        excluded_params = 0
        params_by_layer = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                num_params = param.numel()
                is_excluded = any(layer in name for layer in exclude_layers)
                
                if is_excluded:
                    excluded_params += num_params
                    category = "excluded"
                else:
                    total_params += num_params
                    category = name.split('.')[1] if '.' in name else name
                
                if category not in params_by_layer:
                    params_by_layer[category] = 0
                params_by_layer[category] += num_params

        self.print_v("\nParameter count breakdown:", verbosity=0)
        self.print_v(f"Core model parameters: {total_params:,}", verbosity=0)
        self.print_v(f"Excluded layer parameters: {excluded_params:,}", verbosity=0)

    def save_checkpoint(self, save_name: str, epoch: int, save_to_path: Optional[str] = None) -> None:
        if save_to_path is None:
            if save_name == "BEST":
                save_to_path = os.path.join(self.checkpoint_dir, "BEST")
            elif save_name == "EPOCH":
                save_to_path = os.path.join(self.checkpoint_dir, f"EPOCH{epoch}")
            elif save_name == "LAST":
                save_to_path = os.path.join(self.checkpoint_dir, "LAST")
            else:
                raise Exception(f"Invalid save_name: {save_name}")
        
        if not os.path.exists(save_to_path) and self.accelerator.is_main_process:
            os.makedirs(save_to_path)

        self.print_v(f"Saving checkpoint at epoch {epoch} to {save_to_path}", verbosity=1)
        
        if self.accelerator.is_main_process:
            model_to_save = self.model.module if hasattr(self.model, "module") else self.model
            while hasattr(model_to_save, "module"):
                model_to_save = model_to_save.module
            model_to_save.save_checkpoint(save_to_path)
            torch.save(dict(self.config), os.path.join(save_to_path, "trainer_config.pth"))        

    def save_accelerator_state_to_resume(self, save_name: str, epoch: int) -> None:
        if save_name == "BEST":
            save_to_path = os.path.join(self.checkpoint_dir, "BEST")
        elif save_name == "EPOCH":
            save_to_path = os.path.join(self.checkpoint_dir, f"EPOCH{epoch}")
        elif save_name == "LAST":
            save_to_path = os.path.join(self.checkpoint_dir, "LAST")
        else:
            raise Exception(f"Invalid save_name: {save_name}")
        
        if self.accelerator.is_main_process:
            os.makedirs(save_to_path, exist_ok=True)

        self.print_v(f"Saving accelerator state to {save_to_path}", verbosity=1)
        
        # Each process needs to save its master weights and scheduler+optimizer states.
        # Otherwise, it will hang waiting to synchronize with other processes 
        #            if called just for the process with rank 0.
        self.accelerator.save_state(save_to_path)
        torch.save(epoch, os.path.join(save_to_path, "epoch.pth"))
            
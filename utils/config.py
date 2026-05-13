## YAML CONFIG FILES

import os
import yaml
import argparse

from registry import dataset_defaults, padding_defaults, embedder_defaults

DEFAULT_TRAINER_CONFIG = "configs/trainer.yaml"

def default_trainer_config():
    return update_config(DEFAULT_TRAINER_CONFIG, None)


class DictConfig(dict):

    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = DictConfig(value)
        return value

    def get_dict(self):
        return super()


def unpack_config_rec(config):
    
    # Unpack includes
    if isinstance(config, str) and config.split(":")[0] == "include":
        config = yaml.safe_load(open(config.split(":")[1],"r"))
    
    if isinstance(config, dict):
        for field in config:
            config[field] = unpack_config_rec(config[field])

    return config


def update_config_rec(new_config, config):
    
    if isinstance(config, dict):
        # Force new fields in new_config to update with fields from config
        if not isinstance(new_config,dict):
            print("Created new subdict")
            new_config = {}
        for field in config:
            if not field in new_config:
                # print(f"Creating new field {field}")
                new_config[field] = {}
            new_config[field] = update_config_rec(new_config[field], config[field])
    else:
        # Assign leaf
        new_config = config

    return new_config


def update_config(default_config, config = None):

    if isinstance(default_config, str):
        default_config = yaml.safe_load(open(default_config,"r"))

    # If no config is provided, we iterate using the same config to make sure that the includes
    # are unpacked
    config = default_config if config is None else config

    if isinstance(config, str):
        config = yaml.safe_load(open(config,"r"))

    # Go down the tree to unpack the includes
    unpacked_default_config = unpack_config_rec(default_config)
    unpacked_config = unpack_config_rec(config)

    return DictConfig(update_config_rec(unpacked_default_config, unpacked_config))


def find_key(d, target):
    """Recursively search a nested dict for `target` and return its value."""
    if not isinstance(d, dict):
        return None
    if target in d:
        return d[target]
    for v in d.values():
        result = find_key(v, target)
        if result is not None:
            return result
    return None


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value


def convert_to_dtype(value):

    value = value.strip()

    # Catch list
    if value[0] == "[" and value[-1] == "]":
        value = [convert_to_dtype(v) for v in value[1:-1].split(",")]
    # Catch None
    elif value == "null" or value == "None" or value == "none":
        value = None
    # Catch bool
    elif value == "true" or value == "True":
        value = True
    elif value == "false" or value == "False":
        value = False
    # Catch int
    elif value.isdigit() or value.replace("-","").isdigit():
        value = int(value)
    # Catch float
    else:
        try:
            value = float(value)
        except Exception:   
            pass
    return value
            

def config_from_kwargs(kwargs):
    
    config = {}
    
    if kwargs is not None:
        for key, value in kwargs.items():
            
            # Froms string to aproppriate dtype
            value = convert_to_dtype(value)
            
            # Go iteratively to the leaf
            cur = config
            for sub_key in key.split(".")[:-1]:
                if not sub_key in cur:
                    cur[sub_key] = {}
                cur = cur[sub_key]
            cur[key.split(".")[-1]] = value

    return DictConfig(config)


class ConfigBuilder:
    SUPPORTED_MODES = ("train_from_scratch", "finetune")

    def __init__(self, args):
        self.args = args
        self.config = None
        self.ds_config = None

    def build(self):
        self._load_base_config()
        self._apply_dataset_config()
        self._apply_checkpoint_config()
        self._build_savestring()
        self._load_ds_config()
        return self.config, self.ds_config

    # Load base config
    def _load_base_config(self):
        mode = self.args.training_mode
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"Unsupported training mode: {mode}")

        config_file = f"configs/{mode}/{self.args.task}/{self.args.encoder}/trainer.yaml"
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
        print(f"Loading config from {config_file}")

        self.config = update_config(default_trainer_config(), config_file)
        self.config = update_config(self.config, config_from_kwargs(self.args.kwargs))

    def _load_ds_config(self):
        if self.args.ds_config != "none":
            with open(self.args.ds_config, "r") as f:
                self.ds_config = yaml.safe_load(f)

    # Dataset & model overrides
    def _apply_dataset_config(self):
        args = self.args

        dataset_cfg = dataset_defaults.get(args.dataset)
        padding_cfg = padding_defaults.get(args.dataset)
        embedder_cfg = embedder_defaults.get(args.dataset)
        if embedder_cfg is not None:
            embedder_cfg = embedder_cfg.get(args.encoder)

        if args.features == "all":
            dataset_cfg["features"] = ["tx1", "spikePow"]
            if embedder_cfg and "max_channels" in embedder_cfg:
                embedder_cfg["max_channels"] *= 2
        else:
            dataset_cfg["features"] = [args.features]

        if dataset_cfg:
            self.config["data"]["dataset"].update(dataset_cfg)
            self.config["method"]["dataset_kwargs"]["dataset_name"] = args.dataset
        if padding_cfg:
            self.config["method"]["dataloader_kwargs"]["pad_dict"]["spikes"] = padding_cfg
        if embedder_cfg:
            find_key(self.config["model"], "embedder").update(embedder_cfg)

    # Checkpoint & savestring
    def _apply_checkpoint_config(self):
        ft_ckpt = self._resolve_ckpt(self.args.ft_ckpt)
        if ft_ckpt:
            self._set_model_from_pt(ft_ckpt)

    def _set_model_from_pt(self, path):
        model = self.config["model"]
        encoder = find_key(model, "encoder")
        if encoder is not None:
            encoder["from_pt"] = path
        decoder = find_key(model, "decoder")
        if decoder is not None:
            decoder["from_pt"] = path

    def _build_savestring(self):
        parts = [
            self.args.training_mode,
            self.args.encoder,
            self.args.task,
            self.args.dataset,
            f"feat_{self.args.features}",
        ]
        if self.args.kwargs:
            for k, v in sorted(self.args.kwargs.items()):
                short_key = k.rsplit(".", 1)[-1]
                parts.append(f"{short_key}_{v}")
        self.config["savestring"] = "-".join(parts)

    @staticmethod
    def parse_savestring(savestring):
        _SAVESTRING_KEYS = ("training_mode", "encoder", "task", "dataset", "features")
        parts = savestring.split("-")
        result = dict(zip(_SAVESTRING_KEYS, parts[:5]))
        result["features"] = result["features"].removeprefix("feat_")
        result["kwargs"] = {}
        for part in parts[5:]:
            k, _, v = part.partition("_")
            result["kwargs"][k] = v
        return DictConfig(result)

    @staticmethod
    def _resolve_ckpt(value):
        return None if value == "none" else value

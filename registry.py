
dataset_registry = {
    "brandman_2024_text": "data.brandman_2024_text.prepare_data",
    "willett_2023_text":  "data.willett_2023_text.prepare_data",
}

dataset_defaults = {
    "willett_2023_text": {
        "zscore_block": True,
        "zscore_day": True,
        "area_start": 0,
        "area_end": 128,
        "features": ["tx1"],
    },
    "brandman_2024_text": {
        "zscore_block": False,
        "zscore_day": False,
        "area_start": 0,
        "area_end": 256,
        "features": ["tx1"],
    },
}

padding_defaults = {
    "willett_2023_text": [
        {
          "dim": 0,
          "side": "right",
          "value": -100,
          "truncate": None,
          "min_length": None,
          "max_length": None,
        }, 
    ],
    "brandman_2024_text": [
        {
          "dim": 0,
          "side": "right",
          "value": -100,
          "truncate": None,
          "min_length": None,
          "max_length": None,
        }, 
    ],
}

embedder_defaults = {
    "willett_2023_text": {
        "ndt": {
            "stitch": True,
            "masker": {
                "active": False,
                "ratio": 0.0, 
                "expand_prob": 1.0,
                "max_timespan": 15,
            },
            "max_F": 1250,
            "max_channels": 128,
            "input_dim": 128,
            "pos": False,
            "act": "softsign",
            "bias": True,
            "dropout": 0.2,
        },
    },
    "brandman_2024_text": {
        "ndt": {
            "stitch": True,
            "masker": {
                "active": False,
                "ratio": 0.0, 
                "expand_prob": 1.0,
                "max_timespan": 15, 
            },
            "max_F": 2500,
            "max_channels": 256,
            "input_dim": 256,
            "pos": False,
            "act": "softsign",
            "bias": True,
            "dropout": 0.2,
        },
    },
}

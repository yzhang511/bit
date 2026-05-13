import os
import sys
import torch
import logging
import logging.handlers
import transformers

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.special import gammaln


def plot_gt_pred(gt, pred, epoch=0):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title("Ground Truth")
    im1 = ax1.imshow(gt, aspect="auto", cmap="binary")
    
    ax2.set_title("Prediction")
    im2 = ax2.imshow(pred, aspect="auto", cmap="binary")
    
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)

    fig.suptitle("Epoch: {}".format(epoch))
    return fig

def plot_neurons_r2(gt, pred, epoch=0, neuron_idx=[]):
    # Create one figure and axis for all plots
    fig, axes = plt.subplots(len(neuron_idx), 1, figsize=(12, 5 * len(neuron_idx)))
    r2_values = []  # To store R2 values for each neuron
    
    for neuron in neuron_idx:
        r2 = r2_score(y_true=gt[:, neuron], y_pred=pred[:, neuron])
        r2_values.append(r2)
        ax = axes if len(neuron_idx) == 1 else axes[neuron_idx.index(neuron)]
        ax.plot(gt[:, neuron], label="Ground Truth", color="black")
        ax.plot(pred[:, neuron], label="Prediction", color="royalblue")
        ax.set_title("Neuron: {}, R2: {:.4f}".format(neuron, r2))
        ax.set_xlabel("Time")
        ax.set_ylabel("Rate")
        ax.legend()
    fig.suptitle("Epoch: {}, Avg R2: {:.4f}".format(epoch, np.mean(r2_values)))
    return fig

def neg_log_likelihood(rates, spikes, zero_warning=True):
    """Calculates Poisson negative log likelihood given rates and spikes.
    formula: -log(e^(-r) / n! * r^n)
           = r - n*log(r) + log(n!)

    Parameters
    ----------
    rates : np.ndarray
        numpy array containing rate predictions
    spikes : np.ndarray
        numpy array containing true spike counts
    zero_warning : bool, optional
        Whether to print out warning about 0 rate
        predictions or not

    Returns
    -------
    float
        Total negative log-likelihood of the data
    """
    assert (
            spikes.shape == rates.shape
    ), f"neg_log_likelihood: Rates and spikes should be of the same shape. spikes: {spikes.shape}, rates: {rates.shape}"

    if np.any(np.isnan(spikes)):
        mask = np.isnan(spikes)
        rates = rates[~mask]
        spikes = spikes[~mask]

    assert not np.any(np.isnan(rates)), "neg_log_likelihood: NaN rate predictions found"

    assert np.all(rates >= 0), "neg_log_likelihood: Negative rate predictions found"
    if np.any(rates == 0):
        if zero_warning:
            print(
                "neg_log_likelihood: Zero rate predictions found. Replacing zeros with 1e-9"
            )
        rates[rates == 0] = 1e-9

    result = rates - spikes * np.log(rates) + gammaln(spikes + 1.0)
    return np.sum(result)


def bits_per_spike(rates, spikes):
    """Computes bits per spike of rate predictions given spikes.
    Bits per spike is equal to the difference between the log-likelihoods (in base 2)
    of the rate predictions and the null model (i.e. predicting mean firing rate of each neuron)
    divided by the total number of spikes.

    Parameters
    ----------
    rates : np.ndarray
        3d numpy array containing rate predictions
    spikes : np.ndarray
        3d numpy array containing true spike counts

    Returns
    -------
    float
        Bits per spike of rate predictions
    """
    nll_model = neg_log_likelihood(rates, spikes)
    null_rates = np.tile(
        np.nanmean(spikes, axis=tuple(range(spikes.ndim - 1)), keepdims=True),
        spikes.shape[:-1] + (1,),
    )
    nll_null = neg_log_likelihood(null_rates, spikes, zero_warning=False)
    return (nll_null - nll_model) / np.nansum(spikes) / np.log(2)


def metrics_list(gt, pred, metrics=["bps", "r2"]):

    if np.isnan(gt).any():
        raise ValueError("Ground truth contains NaN values")
    if np.isnan(pred).any():
        raise ValueError("Predictions contain NaN values")
    
    results = {}
    if "bps" in metrics:
        bps_list = []
        for i in tqdm(range(len(gt))): 
            bps = bits_per_spike(gt, pred)
            if np.isinf(bps):
                bps = np.nan
            bps_list.append(bps)
        mean_bps = np.nanmean(bps_list)
        results["bps"] = mean_bps
    elif "r2" in metrics:
        r2_list = []
        for i in tqdm(range(len(gt))): 
            r2 = r2_score(gt.flatten(), pred.flatten())
            if np.isinf(r2):
                r2 = np.nan
            r2_list.append(r2)
        mean_r2 = np.nanmean(r2_list)
        results["r2"] = mean_r2
    return results


def eval_neurons_metric(model, model_inputs, unused_inputs, outputs, config, **kwargs):
    gt = outputs["targets"]

    if config.method.model_kwargs.loss == "mse":
        metric_name = "r2"
        preds = outputs["preds"]
    else:
        metric_name = "bps"
        preds = torch.exp(outputs["preds"])

    mask = (gt == config.method.dataloader_kwargs.pad_dict["spikes"][0]["value"])
    
    gt = gt.masked_fill(mask, 0).float().detach().cpu()
    preds = preds.masked_fill(mask, 0).float().detach().cpu()

    # top k most active neurons
    firing_rates = gt.abs().sum(dim=(0, 1))
    topk = min(1_000, firing_rates.shape[0])
    active_neurons = list(torch.topk(firing_rates, topk).indices)

    results = metrics_list(
        gt = gt.numpy().transpose(-1, 0, 1)[active_neurons], 
        pred = preds.numpy().transpose(-1, 0, 1)[active_neurons], 
        metrics=[metric_name]
    )
    return torch.tensor(results[metric_name], device=model_inputs["spikes"].device)

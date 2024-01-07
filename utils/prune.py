"""
https://github.com/BlackHC/pytorch_datadiet
"""
import torch
import torch.nn.functional as F
from functorch import make_functional_with_buffers, vmap, grad


def compute_el2n_score(unnormalized_model_outputs, targets):
    """
    The EL2N score the error-L2 norm score, ie the Brier score of a single sample with its target (one-hot) label.
    :param unnormalized_model_outputs: BXC the unnormalized model outputs, ie the logits
    :param targets: BxC the one-hot target labels
    :return:
        a tensor of shape B with the EL2N score for each sample
    """
    # compute the softmax of the unnormalized model outputs
    with torch.no_grad():
        softmax_outputs = F.softmax(unnormalized_model_outputs, dim=1)
        # compute the squared L2 norm of the difference between the softmax outputs and the target labels
        # if target is not one-hot, change it to one-hot
        if len(targets.shape) == 1:
          targets = torch.eye(unnormalized_model_outputs.shape[1]).cuda()[targets]
    el2n_score = torch.sum((softmax_outputs - targets) ** 2, dim=1)
    return el2n_score


def compute_grand_score(net, input, target):
    fmodel, params, buffers = make_functional_with_buffers(net)

    fmodel.eval()

    def compute_loss_stateless_model(params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)

        predictions = fmodel(params, buffers, batch)
        loss = F.cross_entropy(predictions, targets)
        return loss

    ft_compute_grad = grad(compute_loss_stateless_model)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

    grad_norms = []

    ft_per_sample_grads = ft_compute_sample_grad(params, buffers, input, target)

    squared_norm = 0
    for param_grad in ft_per_sample_grads:
        squared_norm += param_grad.flatten(1).square().sum(dim=-1)
    grad_norms.append(squared_norm.detach().cpu().numpy() ** 0.5)

    return grad_norms[0]





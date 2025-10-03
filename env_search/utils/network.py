"""Utils for networks."""
import torch


def int_preprocess_onehot(int_envs: torch.Tensor, nc: int) -> torch.Tensor:
    """
    Preprocess int envs for networks.
    Args:
        int_envs: Input int envs (batch_size, env_height, env_width)
        nc: Number of objects

    Returns:
        One-hot encoded and padded envs
    """
    onehot = torch.eye(nc, device=int_envs.device)[
        int_envs.long()]  # (n, env_height, env_width, nc)
    onehot = torch.moveaxis(onehot, 3, 1)  # (n, nc, env_height, env_width)

    return onehot


def int_preprocess(int_envs: torch.Tensor, i_size: int,
                   nc: int, padding: int) -> torch.Tensor:
    """
    Preprocess int envs for networks.
    Args:
        int_envs: Input int envs (batch_size, env_height, env_width)
        i_size: Image size used by the network
        nc: Number of objects
        padding: Int value of the object to use as padding

    Returns:
        One-hot encoded and padded envs
    """
    _, env_height, env_width = int_envs.shape

    onehot = int_preprocess_onehot(int_envs, nc)

    inputs = torch.zeros((len(int_envs), nc, i_size, i_size),
                         device=int_envs.device)
    # Pad the envs with empty tiles.
    inputs[:, padding, :, :] = 1.0
    inputs[:, :, :env_height, :env_width] = onehot
    return inputs


def freeze_params(network):
    for param in network.parameters():
        param.requires_grad = False


def unfreeze_params(network):
    for param in network.parameters():
        param.requires_grad = True

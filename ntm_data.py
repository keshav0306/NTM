import torch

def copy_task_data(num_samples, seq_len, seq_comp_dim):
    # Would give an input array of shape (num_samples, seq_len, seq_comp_dim)
    # and an output array of shape (num_samples, seq_len * 2, seq_comp_dim)
    # the first seq_len of the dim = 1 would be zeros and the rest would be same as the data

    input = torch.randn((num_samples, seq_len, seq_comp_dim))
    output = torch.zeros((num_samples, seq_len * 2, seq_comp_dim))
    output[:, seq_len::, :] = input

    return input, output
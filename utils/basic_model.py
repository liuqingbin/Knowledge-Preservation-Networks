import torch
import numpy as np
from torch import nn
import math
import torch.nn.functional as F

def orth_gru(cell):
    cell.reset_parameters()
    for _, hh, _, _ in cell.all_weights:
        for i in range(0, hh.shape[0], cell.hidden_size):
            torch.nn.init.orthogonal_(hh[i:i + cell.hidden_size], gain=1)
    return cell

# inter attention
class Attention_Vector(nn.Module):
    def __init__(self, hidden_size, dot=False):
        super().__init__()
        if not dot:
            self.attn_w = nn.Linear(hidden_size*2, hidden_size)
            self.attn_u = nn.Linear(hidden_size*2, hidden_size)

            self.attn_v = nn.Parameter(torch.zeros(hidden_size))
            self.attn_v.data.normal_(mean=0, std=1. / math.sqrt(self.attn_v.size(0)))

        self.dot = dot

    def forward(self, hidden, encoder_outputs, mask_metric=None):
        attn_energies = self.score(hidden, encoder_outputs)
        if mask_metric is not None:
            normalized_energy = F.softmax(attn_energies+mask_metric, dim=1)  # [B,1,T]
        else:
            normalized_energy = F.softmax(attn_energies, dim=1)  # [B,1,T]
        context = torch.bmm(encoder_outputs.transpose(1, 2) , normalized_energy.unsqueeze(2)).squeeze(2)  # [1,B,H]
        return attn_energies, normalized_energy, context

    def score(self, hidden, encoder_outputs):
        if self.dot:
            H = torch.unsqueeze(hidden, 2)
            energy = torch.bmm(encoder_outputs, H).squeeze(2)
            return energy
        else:
            H = torch.unsqueeze(hidden, dim=1)
            energy = torch.tanh(self.attn_w(H) + self.attn_u(encoder_outputs))  # [B,T,2H]->[B,T,H]
            energy = torch.bmm(energy, self.attn_v.repeat(encoder_outputs.size(0), 1).unsqueeze(2)).squeeze(2)  # [B,T,1]
            return energy


def list_tensor(_input):
    return torch.from_numpy(np.array(_input))

# utterance 2 tensor
def utt2tensor(sequences, pad_idx=0, use_tile=False, at_least_one=False, is_float=True, pad_tensor = False, pad_dim = 10):
    lengths = [len(seq) for seq in sequences]
    max_lengths, batch_size = max(lengths), len(sequences)
    if (max_lengths == 0) & at_least_one:
        max_lengths = 1
    if pad_tensor:
        max_lengths = pad_dim
    if is_float:
        padded_seqs = torch.ones(batch_size, max_lengths).float()*pad_idx
    else:
        padded_seqs = torch.ones(batch_size, max_lengths).long()*pad_idx
    if use_tile:
        for seq_idx, seq in enumerate(sequences):
            seq+=[seq[-1]]*(max_lengths - lengths[seq_idx])

    for i, seq in enumerate(sequences):
        padded_seqs[i, :lengths[i]] = torch.Tensor(seq)
    return padded_seqs, lengths

# get mask metrics from length
def get_mask_metric(lengths, at_least_one=False, pad_tensor = False, pad_dim = 10):
    max_kbs = max(lengths)
    if pad_tensor:
        max_kbs = pad_dim

    if at_least_one & (max_kbs==0):
        mask_metric = torch.ones(len(lengths), 1).long()
    else:
        mask_metric = torch.ones(len(lengths), max_kbs).long()

    for l_idx, l in enumerate(lengths):
        mask_metric[l_idx, l:] = torch.zeros_like(mask_metric[l_idx, l:])
    return mask_metric

def masked_cross_entropy_for_value(logits, target, pad_idx=0):
    mask = target.ne(pad_idx)
    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = torch.log(logits_flat)
    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    losses = losses * mask.float()
    loss = losses.sum() / (mask.sum().float())
    return loss

def MultinomialKLDivergenceLoss(p_proba, q_proba):  # [B, T, V]
    loss = q_proba * (torch.log(q_proba) - torch.log(p_proba))
    loss = torch.sum(loss)
    if len(p_proba.shape) == 3:
        ave = p_proba.size(1) * p_proba.size(0)
    elif len(p_proba.shape) == 4:
        ave = p_proba.size(2) * p_proba.size(1) * p_proba.size(0)
    else:
        raise ValueError
    return loss / ave
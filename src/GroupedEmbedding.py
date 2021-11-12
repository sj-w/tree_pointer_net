import re
from collections import defaultdict

from torch import nn
import torch
from typing import Optional


class GroupedEmbedding(nn.Embedding):
    __constants__ = ["group_idx"]

    def __init__(
            self,
            grouped_idx: torch.Tensor,
            embedding_dim: int,
            orig_idx: torch.Tensor,
            padding_idx: int,
            max_norm: Optional[float] = float("inf"),
    ):
        if grouped_idx.dim() == 1:
            self.grouped_idx = grouped_idx.unsqueeze(0)
        else:
            self.grouped_idx = grouped_idx
        self.orig_idx = torch.tensor(sorted(orig_idx))
        num_embeddings = int(torch.max(grouped_idx)) + 1
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm)

    def forward(self, input: torch.Tensor):
        gx = self.grouped_idx.expand(input.shape[0], -1).to(input.device)
        emb_input = torch.gather(gx, 1, input)

        return nn.functional.embedding(
            emb_input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse
        )


def generate_group_idx(vocabulary: dict, padding="<pad>"):
    grouped_indices = defaultdict(set)
    orig_indices = set()
    keys_to_group = {x for x in vocabulary.keys() if not re.match(r"ax.*", x) and not re.match(r"<.*>", x)}

    for k in keys_to_group:
        group = k.split('_')[0]
        if group in vocabulary:
            grouped_indices[group].add(vocabulary[k])
            orig_indices.add(vocabulary[group])
        else:
            group += "_0"
            grouped_indices[group].add(vocabulary[k])
            orig_indices.add(vocabulary[group])

    keys_to_keep = vocabulary.keys() - keys_to_group

    for k in keys_to_keep:
        grouped_indices[k].add(vocabulary[k])
        orig_indices.add(vocabulary[k])

    mapping = torch.zeros(len(vocabulary), dtype=torch.long)
    padding_idx = None

    for g_idx, (k, idxs) in enumerate(sorted(grouped_indices.items(), key=lambda x: vocabulary[x[0]])):
        if k == padding:
            padding_idx = g_idx

        for i in idxs:
            mapping[i] = g_idx

    return mapping, orig_indices, padding_idx


if __name__ == "__main__":
    vocabulary = {'<s>': 0, '<pad>': 1, '</s>': 2, '<unk>': 3, '(': 4, ')': 5, '*': 6, '0': 7, '+': 8, '1': 9, '**': 10,
            '*_0': 11, '<SEP>': 12, '+_0': 13, '**_0': 14, '0_0': 15, 'x': 16, '*_1': 17, '1_0': 18, '+_1': 19, '0_1': 20,
            'x_0': 21, '-': 22, 'z': 23, '*_2': 24, 'y': 25, '**_1': 26, '1_1': 27, '-_0': 28, '2': 29, '-1': 30, '3': 31,
            '0_2': 32, '-1_0': 33, 'x_1': 34, '+_2': 35, 'z_0': 36, '2_0': 37, 'y_0': 38, '*_3': 39, '3_0': 40, '4': 41,
            '1_2': 42, '**_2': 43, '0_3': 44, '4_0': 45, '*_4': 46, '+_3': 47, '-1_1': 48, '-_1': 49, 'y_1': 50, 'z_1': 51,
            '-4': 52, 'x_2': 53, 'ax_39': 54, 'ax_40': 55, 'ax_41': 56, '1_3': 57, '0_4': 58, '-4_0': 59, '*_5': 60,
            '**_3': 61, '+_4': 62, '2_1': 63, 'ax_16': 64, 'ax_15': 65, 'ax_29': 66, 'ax_19': 67, '3_1': 68, 'ax_18': 69,
            'ax_37': 70, '*_6': 71, 'x_3': 72, '4_1': 73, '0_5': 74, '1_4': 75, '-_2': 76, '-2': 77, '-2_0': 78, '+_5': 79,
            '**_4': 80, 'z_2': 81, 'y_2': 82, '-3': 83, '-4_1': 84, 'ax_38': 85, '-1_2': 86, '*_7': 87, 'ax_21': 88,
            '0_6': 89, 'ax_23': 90, 'ax_27': 91, 'ax_25': 92, '2_2': 93, 'ax_1': 94, 'z_3': 95, 'y_3': 96, 'ax_36': 97,
            '-3_0': 98, 'ax_31': 99}

    grouped_idx, orig_idx, padding_idx = generate_group_idx(vocabulary)
    embedding = GroupedEmbedding(grouped_idx, 64, padding_idx)
    x = torch.LongTensor([[8,9,1,5,19]])
    foo = embedding(x)
    print()

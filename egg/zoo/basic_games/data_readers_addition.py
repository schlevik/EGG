# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch.utils.data import Dataset


class SumGameDataset(Dataset):
    # adapted from data_readers.py
    def __init__(self, path, n_max, n_summands=2):
        frame = np.loadtxt(path, dtype="S10")
        self.frame = []
        for row in frame:
            config = [int(r) for r in row]
            z = torch.zeros(n_max * n_summands)
            assert len(config) == n_summands
            for i in range(n_summands):
                # set n-th entry to 1 with offset i <- 0..n_summands
                # corresponds to the the concatenation of 2 1-hot encodings
                z[config[i] + i * n_max] = 1
            label = torch.tensor(sum(config))
            self.frame.append((z.view(-1), label))

    def get_n_features(self):
        return self.frame[0][0].size(0)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return self.frame[idx]

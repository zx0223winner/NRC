"""
Authors: Nicolas Raymond

Description: Creation of a heatmap in which each squares
             represent the impact of replacing a single nucleotide
             for another one.
"""

import sys
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import argmax, arange, array, concatenate, unravel_index
from os.path import abspath, join, pardir
from torch import device, load

# Project imports
sys.path.append(abspath(join(__file__, *[pardir] * 2)))
from src.models.cnn import WCNN
from src.utils.analysis import MASK_VALUE
from src.utils.gene_editing import predict_editing_impact
from src.data.modules.preprocessing import nuc_to_proba

# Set constants
BATCH_SIZE: int = 1000
MODEL_PATH: str = 'path_to_pt_file'

# Execution of the script
if __name__ == '__main__':
    model = WCNN(regression=True, encoding_size=4)
    model.load_state_dict(load(MODEL_PATH))

    seq = input('\nEnter the sequence: ').upper()
    print(len(seq))
    encoded_seq = nuc_to_proba(seq=seq).permute(1, 0)

    # Predict the percentage of expression difference between
    # the sequence and all of its possible single-base variations
    mrna_diff = predict_editing_impact(model=model,
                                       seq=encoded_seq.unsqueeze(dim=0),
                                       dev=device('cuda'),
                                       batch_size=BATCH_SIZE).numpy()

    # We generate the figure
    plt.rcParams.update({'font.size': 5, 'font.family': 'serif'})
    plt.ticklabel_format(style='plain')
    fig, ax = plt.subplots(figsize=(5, 1))
    start, stop = 975, 1026
    arg_max = unravel_index(argmax(mrna_diff[:, start:stop], axis=None),
                            mrna_diff[:, start:stop].shape)
    print(f'Argmax = {arg_max}')
    sns.heatmap(mrna_diff[:, start:stop], ax=ax, linewidths=0.2, square=True, cmap='viridis',
                cbar_kws=dict(use_gridspec=False,
                              location='top',
                              fraction=0.22,
                              aspect=120,
                              shrink=0.45,
                              label='Predicted effect size'),
                mask=(mrna_diff[:, start:stop] == MASK_VALUE))
    xticks = concatenate([arange(0.5, (1000 - start) + 0.5, 5),
                          array([25.5]), arange(30.5, (stop - start) + 0.5, 5)])
    ax.set_xticks(xticks)
    ax.set_xticklabels(list(range(start, 1000, 5)) + ['TSS'] + list(range(1005, stop, 5)))
    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_yticklabels(['A', 'C', 'G', 'T'])
    ax.tick_params(axis='y', width=0.2, labelrotation=45, length=1)
    ax.tick_params(axis='x', width=0.2, labelrotation=0, pad=1.5, length=1)
    ax.set_xlabel('Nucleotide position')
    ax.set_ylabel('Nucleotide')
    plt.savefig('single_base_editing.pdf')

"""
Authors: Nicolas Raymond

Description: Stores the iterative algorithm aiming to find the single-base edits
             that maximizes the expression level of a gene.
"""

from src.data.modules.constants import Maps
from src.utils.analysis import MASK_VALUE, save_editing_heatmap
from torch import argmax, autocast, cat, device, float16, no_grad, Tensor
from torch.nn import Module

VARIATIONS_NB: int = 12000


def get_editing_strategy(model: Module,
                         seq: Tensor,
                         dev: device,
                         max_edits: int = 1,
                         batch_size: int = 1,
                         save_figs: bool = True) -> list[tuple[str, int, int]]:
    """
    Generates a list of singe-base edits to increase the expression level of a gene.

    Args:
        model (Module): Instance of a torch module.
        seq (Tensor): One-hot encoded sequence of shape (1, 5, 3000) or (5, 3000)
        dev (device): torch device.
        max_edits (int, optional): maximum number of singe-base edits available.
                                   Default to 1.
        batch_size (int, optional): Batch size used for the forward pass. Default to 1.
        save_figs (bool, optional): if True, figures are saved at each iteration to
                                    visualize the edit.
                                    Default to True.


    Returns:
        tuple[list[tuple[str, int, int]]: list of tuples showing the edits to be made,
                                          their location and the expected improvement
                                          associated to them.
    """
    # Check the shape of the input sequence
    if len(seq.shape) == 2:
        seq = seq.unsqueeze(dim=0)
    if seq.shape[0] != 1 or seq.shape[1] != 4 or seq.shape[2] != 3000:
        raise ValueError('Provided sequence must be of shape (1, 4, 3000) or (4, 3000).')

    # Initialize a list containing the edits to make
    edits = []

    # For each iteration until the reach of the maximal budget of edits
    for i in range(max_edits):

        # Predict the impact of each possible single-base modification
        pred = predict_editing_impact(model=model,
                                      seq=seq,
                                      dev=dev,
                                      batch_size=batch_size)

        # Find the single-base modification providing the greatest increase
        arg_max = argmax(pred)
        x, y = (arg_max // pred.shape[1]).item(), (arg_max % pred.shape[1]).item()

        # If the maximal increase found is negative or equal to 0, we stop the loop
        if pred[x, y] <= 0:
            print(f'Algorithm stopped prematurely at iteration {i} because no more ' +
                  'edits were found to improve transcript abundance')
            break

        else:

            # Save the edit
            new_nucleotide = Maps.PROBA2NUC[x]
            old_nucleotide = Maps.PROBA2NUC[seq[0, :, y].nonzero().item()]
            edits.append((f'{old_nucleotide}->{new_nucleotide}', y, round(pred[x, y].item(), 3)))

            # Modify the sequence
            seq[0, :, y] = Tensor(Maps.NUC2PROBA[new_nucleotide])

            if save_figs:
                save_editing_heatmap(predictions=pred.numpy(), arg_max=y, iteration=i)

    return edits


def generate_sequence_variations(seq: Tensor) -> Tensor:
    """
    Generate all variations of a sequence that can be obtained by replacing a single base.

    Args:
        seq (Tensor): soft encoded sequence (1, 4, 3000)

    Returns:
        Tensor: all variations of the sequence (12000, 4, 3000)
    """
    variations = cat([seq.clone().detach() for _ in range(VARIATIONS_NB)])
    for i in range(4):
        for j in range(3000):
            variations[(i * 3000) + j, :, j] = 0
            variations[(i * 3000) + j, i, j] = 1

    return variations


def predict_editing_impact(model: Module,
                           seq: Tensor,
                           dev: device,
                           batch_size: int = 1) -> Tensor:
    """
    Predict the mRNA abundance difference between all single-base
    variations of a sequence and the sequence itself.

    Please be aware that the batch size can slightly affect the embeddings
    generated (see https://github.com/huggingface/transformers/issues/2401).

    Args:
        model (Module): Instance of the PlanTT architecture.
        seq (Tensor): Soft encoded sequence of shape (1, 4, 3000)
        dev (device): torch device.
        batch_size (int, optional): Batch size used for the forward pass. Default to 1.
        epsilon (float, optional): Small value added to the denominator to avoid zero division.
                                   Default to 1e-4.

    Returns:
        Tensor: Predictions of the percentage of expression difference predicted for all variations.
                The output is a tensor of shape (4, 3000) where each row is respectively
                associated to nucleotide ACGT and each column represents a position in
                the 3kb long sequence. For example, the element at position (0, 1200)
                represents the impact of having an A at the position 1200.
    """
    # Generate all sequence variations
    variations = generate_sequence_variations(seq)

    # Set model in eval model and send to device
    model.eval()
    model.to(dev)

    with no_grad():
        start_idx = 0
        predictions = []

        if dev.type == 'cuda':
            while start_idx < VARIATIONS_NB:
                batch = variations[start_idx:(start_idx + batch_size)].to(dev)
                seq_tiled = seq.repeat(batch.shape[0], 1, 1)

                with autocast(device_type=dev.type, dtype=float16):
                    predictions.append(model(seq_tiled.to(dev), batch).to('cpu'))
                start_idx += batch_size

        # Reshape the predictions in a grid of shape (4, 3000) where each
        # row is respectively associated to nucleotides ACGT and each column
        # represents a position in the 3kb long sequence.
        predictions = cat(predictions).reshape(4, 3000)

        # Manually set coordinates where the predictions must have been equal to zero
        # Some predictions are slightly different than 0 due to batch size effect on GPU
        # (see https://github.com/huggingface/transformers/issues/2401)
        zero_coord = (seq[0, :, :] == 1).nonzero(as_tuple=True)
        predictions[zero_coord[0], zero_coord[1]] = MASK_VALUE

        return predictions

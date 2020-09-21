from fractions import Fraction
import pdb

import torch
from torch.utils.data import DataLoader

def eval(model, test_data):
    model.eval()
    total_count = 0
    correct_count = 0

    with torch.no_grad():
        for inputs, expected_outputs in test_data:
            assert inputs.shape == expected_outputs.shape, f'inputs:{inputs.shape} outputs:{expected_outputs}'
            batch_size = expected_outputs.shape[0]
            total_count += batch_size

            logits = model(inputs.type(torch.get_default_dtype()))
            actual_outputs = logits > 0
            assert actual_outputs.shape == expected_outputs.shape, f'expected:{expected_outputs.shape} actual:{actual_outputs.shape}'
            is_same = (expected_outputs == actual_outputs).view(batch_size, -1).all(1)
            correct_count += is_same.sum(dtype=torch.int).item()

    return Fraction(correct_count, total_count, _normalize=False)

from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from .eval import eval

def fit(model, train_data, valid_datasets, early_stop=None, display_freq=100, valid_freq=1000):
    optimizer = optim.Adam(model.parameters())
    loss_func = nn.BCEWithLogitsLoss()

    best_valid_accuracy = 0.0
    stalled_step = 0

    for train_step, (train_inputs, train_outputs) in enumerate(train_data):
        assert train_inputs.shape == train_outputs.shape, f'inputs:{train_inputs.shape} outputs:{train_outputs}'
        model.train()

        optimizer.zero_grad()
        logits = model(train_inputs.type(torch.get_default_dtype()))
        assert train_outputs.shape == logits.shape, f'expected:{train_outputs.shape} actual:{logits.shape}'

        loss = loss_func(logits, train_outputs.type_as(logits))
        loss.backward()
        optimizer.step()

        step_count = train_step + 1
        if step_count % display_freq == 0:
            print(f'{datetime.now()} step {step_count} loss {loss.item()}')

        if len(valid_datasets) > 0 and (step_count) % valid_freq == 0:
            print(f'Validation on step {step_count}...')
            for i, (name, valid_data) in enumerate(valid_datasets):
                accuracy = eval(model, valid_data)
                print(f'Accuracy on {name} is {accuracy}')

                if i == 0:
                    if accuracy > best_valid_accuracy:
                        print(f'New best! (last best {best_valid_accuracy})')
                        best_valid_accuracy = accuracy
                        stalled_step = 0
                    else:
                        stalled_step += 1
                        print(f'Stalled for {stalled_step} steps')
            if stalled_step == early_stop:
                break

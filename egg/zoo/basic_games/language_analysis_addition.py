import json
from itertools import count
from typing import List, Tuple

import numpy as np
import tabulate

from egg.core import Callback, Interaction


def to_numbers(array: List[int]) -> Tuple[int, int]:
    x = array[len(array) // 2:]
    y = array[:len(array) // 2]
    return int(np.argmax(x)), int(np.argmax(y))


class PrintValidationEventsForAddition(Callback):
    """
    Adopted from core/language_analysis.py
    """
    def __init__(self, n_epochs, save_to=None):
        super().__init__()
        self.n_epochs = n_epochs # how often to print save
        self.save_to = save_to # where to save

    @staticmethod
    def print_events(logs: Interaction, save_to=None):
        input_pairs = [to_numbers(m.tolist()) for m in logs.sender_input]
        encoded_inputs = [m.tolist() for m in logs.sender_input]
        labels = [m.tolist() for m in logs.labels]
        messages = [m.tolist() for m in logs.message]
        full_outputs = [m.tolist() for m in logs.receiver_output]
        predicted_output = [int(np.argmax(m)) for m in full_outputs]
        correct = [bool(p == l) for p, l in zip(predicted_output, labels)]
        headers = ['input', 'label', 'message', "output", "correct"]
        data = list(zip(input_pairs, labels, messages, predicted_output, correct))

        # nicer printing of the outputs
        print(tabulate.tabulate(data, headers=headers))
        if save_to:
            dicted_data = [{k: v for k, v in zip(headers, d)} for d in data]
            with open(save_to, "w+") as f:
                f.write('\n'.join(json.dumps(d) for d in dicted_data) + '\n')

    # here is where we make sure we are printing the validation set (on_validation_end, not on_epoch_end)
    def on_validation_end(self, _loss, logs: Interaction, epoch: int):
        # here is where we check that we are at the last epoch
        if epoch == self.n_epochs:
            self.print_events(logs, self.save_to)

    # same behaviour if we reached early stopping
    def on_early_stopping(self, _train_loss, _train_logs, epoch, _test_loss, test_logs):
        self.print_events(test_logs)


class StoreEvaluationScoreCallback(Callback):
    # this is just something to keep the results of the evaluation
    def __init__(self, n_epochs):
        super().__init__()
        self.n_epochs = n_epochs

        self.val_loss = None
        self.acc = None

    def on_validation_end(self, _loss, logs: Interaction, epoch: int):
        # here is where we check that we are at the last epoch
        if epoch == self.n_epochs:
            self.val_loss = _loss
            self.acc = logs.aux['acc'].mean().item()

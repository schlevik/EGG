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
    def __init__(self, n_epochs):
        super().__init__()
        self.n_epochs = n_epochs

    @staticmethod
    def print_events(logs: Interaction):
        input_pairs = [to_numbers(m.tolist()) for m in logs.sender_input]
        encoded_inputs = [m.tolist() for m in logs.sender_input]
        labels = [m.tolist() for m in logs.labels]
        messages = [m.tolist() for m in logs.message]
        full_outputs = [m.tolist() for m in logs.receiver_output]
        predicted_output = [np.argmax(m) for m in full_outputs]
        correct = [p == l for p,l in zip(predicted_output, labels)]
        print(tabulate.tabulate(zip(input_pairs, labels, messages, predicted_output),
                                headers=['Inputs', 'Labels', 'Messages', "Outputs", "Correct?"]))

    # here is where we make sure we are printing the validation set (on_validation_end, not on_epoch_end)
    def on_validation_end(self, _loss, logs: Interaction, epoch: int):
        # here is where we check that we are at the last epoch
        if epoch == self.n_epochs:
            self.print_events(logs)

    # same behaviour if we reached early stopping
    def on_early_stopping(self, _train_loss, _train_logs, epoch, _test_loss, test_logs):
        self.print_events(test_logs)

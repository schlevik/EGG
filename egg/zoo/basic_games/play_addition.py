# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
from collections import defaultdict
from copy import deepcopy
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from ax.service.ax_client import AxClient
from torch.utils.data import DataLoader

import egg.core as core
from egg.core import Callback, Interaction, PrintValidationEvents
from egg.core.baselines import BuiltInBaseline
from egg.core.callbacks import WandbLogger
from egg.zoo.basic_games.architectures import DiscriReceiver, RecoReceiver, Sender, FixedLengthSender, \
    FixedLengthReceiver
from egg.zoo.basic_games.data_readers import AttValDiscriDataset, AttValRecoDataset

# the following section specifies parameters that are specific to our games: we will also inherit the
# standard EGG parameters from https://github.com/facebookresearch/EGG/blob/master/egg/core/util.py
from egg.zoo.basic_games.data_readers_addition import SumGameDataset
from egg.zoo.basic_games.language_analysis_addition import PrintValidationEventsForAddition, \
    StoreEvaluationScoreCallback


def get_params(params):
    parser = argparse.ArgumentParser()
    # arguments controlling the game type
    # arguments concerning the input data and how they are processed
    parser.add_argument(
        "--train_data", type=str, default=None, help="Path to the train data"
    )
    parser.add_argument(
        "--validation_data", type=str, default=None, help="Path to the validation data"
    )
    # (the following is only used in the reco game)
    parser.add_argument(
        "--n_max",
        type=int,
        help="Highest input number of DS",
    )
    parser.add_argument(
        "--depth_sender",
        type=int,
        default=2,
        help="Number of layers for sender/receiver",
    )
    parser.add_argument(
        "--depth_receiver",
        type=int,
        default=2,
        help="Number of layers for sender/receiver",
    )
    parser.add_argument(
        "--n_summands",
        type=int,
        default=2,
        help="Number of summands to add",
    )
    parser.add_argument(
        "--validation_batch_size",
        type=int,
        default=0,
        help="Batch size when processing validation data, whereas training data "
             "batch_size is controlled by batch_size (default: same as training data batch size)",
    )
    # arguments concerning the training method
    parser.add_argument(
        "--mode",
        type=str,
        default="rf",
        help="Selects whether Reinforce or Gumbel-Softmax relaxation is used for training {rf, gs} (default: rf)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="GS temperature for the sender, only relevant in Gumbel-Softmax (gs) mode (default: 1.0)",
    )
    parser.add_argument(
        "--sender_entropy_coeff",
        type=float,
        default=1e-1,
        help="Reinforce entropy regularization coefficient for Sender, only"
             " relevant in Reinforce (rf) mode (default: 1e-1)",
    )
    # arguments concerning the agent architectures
    parser.add_argument(
        "--sender_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--receiver_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--sender_hidden",
        type=int,
        default=10,
        help="Size of the hidden layer of Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=10,
        help="Size of the hidden layer of Receiver (default: 10)",
    )
    parser.add_argument(
        "--sender_embedding",
        type=int,
        default=10,
        help="Output dimensionality of the layer that embeds symbols produced at previous step in Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_embedding",
        type=int,
        default=10,
        help="Output dimensionality of the layer that embeds the message symbols for Receiver (default: 10)",
    )
    # arguments controlling the script output
    parser.add_argument(
        "--print_validation_events",
        default=False,
        action="store_true",
        help="If this flag is passed, at the end of"
             "training the script prints the input validation data, the corresponding messages produced by the Sender, "
             "and the output probabilities produced by the Receiver (default: do not print)",
    )
    parser.add_argument(
        "--log_every", type=int, default=1, help="How often to log train/eval data"
    )
    parser.add_argument(
        "--num_heads_sender",
        type=int,
        default=8,
        help="Output dimensionality of the layer that embeds symbols produced at previous step in Sender (default: 10)",
    )
    parser.add_argument(
        "--num_layers_sender",
        type=int,
        default=12,
        help="Output dimensionality of the layer that embeds symbols produced at previous step in Sender (default: 10)",
    )
    parser.add_argument(
        "--num_heads_receiver",
        type=int,
        default=8,
        help="Output dimensionality of the layer that embeds symbols produced at previous step in Sender (default: 10)",
    )
    parser.add_argument(
        "--num_layers_receiver",
        type=int,
        default=12,
        help="Output dimensionality of the layer that embeds symbols produced at previous step in Sender (default: 10)",
    )
    parser.add_argument(
        "--proj_name",
        type=str,
        default='egg',
        help="Name of project",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name of run",
    )
    parser.add_argument(
        "--sweep_mode",
        action='store_true',
        default=False,
        help="Whether this is a wandb hyper-param sweep",
    )

    parser.add_argument(
        "--ax_hp",
        type=str,
        default=None,
        help='Ax hyper-parameter file (json).'
    )
    args = core.init(parser, params)
    return args


def main(opts):
    if opts.validation_batch_size == 0:
        opts.validation_batch_size = opts.batch_size

    wandb.init(config=opts, project=opts.proj_name, name=opts.run_name)
    opts = wandb.config

    print(opts, flush=True)

    # the following if statement controls aspects specific to the two game tasks: loss, input data and architecture of the Receiver
    # (the Sender is identical in both cases, mapping a single input attribute-value vector to a variable-length message)

    def loss(
            sender_input, _message, _receiver_input, receiver_output, labels, _aux_input, do_reduce=False
    ):
        # in the case of the recognition game, for each attribute we compute a different cross-entropy score
        # based on comparing the probability distribution produced by the Receiver over the values of each attribute
        # with the corresponding ground truth, and then averaging across attributes
        # accuracy is instead computed by considering as a hit only cases where, for each attribute, the Receiver
        # assigned the largest probability to the correct value
        # most of this function consists of the usual pytorch madness needed to reshape tensors in order to perform these computations
        n_attributes = opts.n_max
        n_values = opts.n_summands
        batch_size = sender_input.size(0)
        l = F.cross_entropy(receiver_output, labels, reduction='none' if not do_reduce else 'mean')
        receiver_guesses = receiver_output.argmax(dim=1)
        acc = (receiver_guesses == labels).float()
        # receiver_output = receiver_output.view(batch_size * n_attributes, n_values)
        # receiver_guesses = receiver_output.argmax(dim=1)
        # correct_samples = (
        #     (receiver_guesses == labels.view(-1))
        #         .view(batch_size, n_attributes)
        #         .detach()
        # )
        # acc = (torch.sum(correct_samples, dim=-1) == n_attributes).float()
        # labels = labels.view(batch_size * n_attributes)
        # loss = F.cross_entropy(receiver_output, labels, reduction="none")
        # loss = loss.view(batch_size, -1).mean(dim=1)
        return l, {"acc": acc}

    # again, see data_readers.py in this directory for the AttValRecoDataset data reading class
    train_loader = DataLoader(
        SumGameDataset(
            path=opts.train_data,
            n_max=opts.n_max
        ),
        batch_size=opts.batch_size,
        shuffle=True,
        num_workers=1,
    )
    test_loader = DataLoader(
        SumGameDataset(
            path=opts.validation_data,
            n_max=opts.n_max
        ),
        batch_size=opts.validation_batch_size,
        shuffle=False,
        num_workers=1,
    )
    # the number of features for the Receiver (input) and the Sender (output) is given by n_attributes*n_values because
    # they are fed/produce 1-hot representations of the input vectors
    n_features = opts.n_max * opts.n_summands
    # we define here the core of the receiver for the discriminative game, see the architectures.py file for details
    # this will be embedded in a wrapper below to define the full architecture
    receiver = RecoReceiver(n_features=n_features,
                            n_hidden=opts.receiver_hidden) if opts.max_len > 1 else FixedLengthReceiver(
        n_features=n_features, n_hidden=opts.receiver_hidden, depth=opts.depth_receiver)

    # we are now outside the block that defined game-type-specific aspects of the games: note that the core Sender architecture
    # (see architectures.py for details) is shared by the two games (it maps an input vector to a hidden layer that will be use to initialize
    # the message-producing RNN): this will also be embedded in a wrapper below to define the full architecture
    sender = Sender(n_hidden=opts.sender_hidden, n_features=n_features) if opts.max_len > 1 else FixedLengthSender(
        n_hidden=opts.sender_hidden, n_features=n_features, vocab_size=opts.vocab_size, depth=opts.depth_receiver)

    # now, we instantiate the full sender and receiver architectures, and connect them and the loss into a game object
    # the implementation differs slightly depending on whether communication is optimized via Gumbel-Softmax ('gs') or Reinforce ('rf', default)
    if opts.mode.lower() == "gs":
        # in the following lines, we embed the Sender and Receiver architectures into standard EGG wrappers that are appropriate for Gumbel-Softmax optimization
        # the Sender wrapper takes the hidden layer produced by the core agent architecture we defined above when processing input, and uses it to initialize
        # the RNN that generates the message
        if opts.max_len == 1:
            sender = core.GumbelSoftmaxWrapper(sender)

            receiver = core.SymbolReceiverWrapper(receiver, vocab_size=opts.vocab_size,
                                                  agent_input_size=opts.receiver_hidden)
            game = core.SymbolGameGS(sender, receiver, lambda *args, **kwargs: loss(*args, **kwargs, do_reduce=False))
        else:

            sender = core.RnnSenderGS(
                sender,
                vocab_size=opts.vocab_size,
                embed_dim=opts.sender_embedding,
                hidden_size=opts.sender_hidden,
                cell=opts.sender_cell,
                max_len=opts.max_len,
                temperature=opts.temperature,
            )
            # the Receiver wrapper takes the symbol produced by the Sender at each step (more precisely, in Gumbel-Softmax mode, a function of the overall probability
            # of non-eos symbols upt to the step is used), maps it to a hidden layer through a RNN, and feeds this hidden layer to the
            # core Receiver architecture we defined above (possibly with other Receiver input, as determined by the core architecture) to generate the output
            receiver = core.RnnReceiverGS(
                receiver,
                vocab_size=opts.vocab_size,
                embed_dim=opts.receiver_embedding,
                hidden_size=opts.receiver_hidden,
                cell=opts.receiver_cell,
            )
            game = core.SenderReceiverRnnGS(sender, receiver, loss)
        # callback functions can be passed to the trainer object (see below) to operate at certain steps of training and validation
        # for example, the TemperatureUpdater (defined in callbacks.py in the core directory) will update the Gumbel-Softmax temperature hyperparameter
        # after each epoch
        callbacks = [core.TemperatureUpdater(agent=sender, decay=0.9, minimum=0.1)]
    else:  # NB: any other string than gs will lead to rf training!
        # here, the interesting thing to note is that we use the same core
        # architectures we defined above, but now we embed them in wrappers that are suited to
        # Reinforce-based optmization

        if opts.max_len == 1:
            sender = core.ReinforceWrapper(sender)

            receiver = core.SymbolReceiverWrapper(receiver, vocab_size=opts.vocab_size,
                                                  agent_input_size=opts.receiver_hidden)
            game = core.SymbolGameReinforce(sender, receiver,
                                            lambda *args, **kwargs: loss(*args, **kwargs, do_reduce=False),
                                            sender_entropy_coeff=opts.sender_entropy_coeff,
                                            receiver_entropy_coeff=0)
        else:
            if opts.sender_cell == 'tf':
                sender = Sender(n_features=n_features, n_hidden=opts.sender_embedding)
                sender = core.TransformerSenderReinforce(
                    sender,
                    vocab_size=opts.vocab_size,
                    embed_dim=opts.sender_embedding,
                    max_len=opts.max_len,
                    num_layers=opts.num_layers_sender,
                    num_heads=opts.num_heads_sender,
                    hidden_size=opts.sender_hidden,
                )
            else:
                sender = core.RnnSenderReinforce(
                    sender,
                    vocab_size=opts.vocab_size,
                    embed_dim=opts.sender_embedding,
                    hidden_size=opts.sender_hidden,
                    cell=opts.sender_cell,
                    max_len=opts.max_len,
                )
            if opts.receiver_cell == 'tf':
                receiver = RecoReceiver(
                    n_features, n_hidden=opts.receiver_embedding
                )
                # sender = Sender(n_features=n_features, n_hidden=opts.sender_embedding)
                receiver = core.TransformerReceiverDeterministic(
                    receiver,
                    vocab_size=opts.vocab_size,
                    max_len=opts.max_len,
                    embed_dim=opts.receiver_embedding,
                    num_heads=opts.num_heads_receiver,
                    hidden_size=opts.receiver_hidden,
                    num_layers=opts.num_layers_receiver,
                    causal=False,
                )
            else:

                receiver = core.RnnReceiverDeterministic(
                    receiver,
                    vocab_size=opts.vocab_size,
                    embed_dim=opts.receiver_embedding,
                    hidden_size=opts.receiver_hidden,
                    cell=opts.receiver_cell,
                )
            game = core.SenderReceiverRnnReinforce(
                sender,
                receiver,
                lambda *args, **kwargs: loss(*args, **kwargs, do_reduce=False),
                sender_entropy_coeff=opts.sender_entropy_coeff,
                receiver_entropy_coeff=0,
            )
        callbacks: List[Callback] = []

    # we are almost ready to train: we define here an optimizer calling standard pytorch functionality
    optimizer = core.build_optimizer(game.parameters())
    # in the following statement, we finally instantiate the trainer
    # object with all the components we defined (the game, the optimizer, the data
    # and the callbacks)
    res = StoreEvaluationScoreCallback(n_epochs=opts.n_epochs)
    if opts.print_validation_events == True:
        # we add a callback that will print loss and accuracy after each
        # training and validation pass (see ConsoleLogger in callbacks.py in core directory)
        # if requested by the user, we will also print a detailed log of the
        # validation pass after full training: look at PrintValidationEvents in
        # language_analysis.py (core directory)
        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_loader,
            validation_data=test_loader,
            callbacks=callbacks
                      + [
                          WandbLogger(opts, opts.proj_name, opts.run_name, sweep_mode=opts.sweep_mode),
                          res if opts.ax_hp else core.ConsoleLogger(print_train_loss=True, as_json=True, every_x=10),
                          PrintValidationEventsForAddition(n_epochs=opts.n_epochs),
                      ],
        )
    else:
        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_loader,
            validation_data=test_loader,
            callbacks=callbacks
                      + [
                          WandbLogger(opts, opts.proj_name, opts.run_name),
                          res if opts.ax_hp else core.ConsoleLogger(print_train_loss=True, as_json=True, every_x=10),
                      ],
        )

    # and finally we train!
    trainer.train(n_epochs=opts.n_epochs)
    return res.acc, res.val_loss
    # trainer.eval()


def run_hpopt(train_and_eval_single_step: callable, hp_dict, opts):
    # number of trials of different combinations
    hyperparam_opt_runs = hp_dict['num_runs']

    # number of runs for the same combination (results are averaged)
    #runs_per_trial = hp_dict['runs_per_trial']

    ax_client = AxClient(verbose_logging=False)
    ax_client.create_experiment(
        name=f'nnl',
        parameters=hp_dict['hyper_params'],
        objective_name='acc',  # what we maximise
        minimize=False,
    )

    # first, eval and save what is the performance before training
    hpopt_results = defaultdict(dict)
    # run hyperparam optimisation
    for i in range(hyperparam_opt_runs):

        parameters, trial_index = ax_client.get_next_trial()
        single_step_args = deepcopy(opts)
        # update the params with what ax wants to try next
        for k, v in parameters.items():
            setattr(single_step_args, k, v)
        print(f"Trying parameters: {parameters}")
        # perform runs_per_trial training/evaluation steps and obtain the mean of the metric
        # we want to optimise (mcc)
        acc, loss = train_and_eval_single_step(single_step_args)
        hpopt_results[i]['tried_params'] = parameters
        hpopt_results[i]['result'] = {"acc": acc, "loss": loss}

        print(f"Result: {acc} (loss: {loss:.4f})")
        # update ax
        ax_client.complete_trial(trial_index=trial_index, raw_data=acc)
    best_params, metrics = ax_client.get_best_parameters()
    print(best_params)
    print(metrics)
    with open(opts.ax_hp.replace(".json", '-results.json'), 'w+') as f:
        json.dump({"best_params": best_params, "hpopt_results": hpopt_results, "metrics": metrics}, f, indent=2)


if __name__ == "__main__":
    import sys

    opts = get_params(sys.argv[1:])

    if opts.ax_hp is not None:
        os.environ['WANDB_MODE'] = 'disabled'
        with open(opts.ax_hp) as f:
            hp_dict = json.load(f)
            run_hpopt(main, hp_dict, opts)
    else:
        main(opts)

# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
# Code adapted from transfer-learning-conv-ai repo of Hugging Face
import os
import math
import random
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                  GPT2DoubleHeadsModel, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME)

from utils import get_dataset, make_logdir
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", '<bof>', "<pad>", "<bor>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>', '<bof>', '<bor>']}
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

logger = logging.getLogger(__file__)

def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def pad_dataset(dataset, padding=0):
    """ Pad the dataset. This could be optimized by defining a Dataset class and padding at the batch level, but this is simpler. """
    max_l = max(len(x) for x in dataset["input_ids"])
    for name in PADDED_INPUTS:
        dataset[name] = [x + [padding if name != "lm_labels" else -100] * (max_l - len(x)) for x in dataset[name]]
    return dataset


def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    frame_word_list = []
    with open('frames_list.txt', encoding='utf-8') as f:
        for sent in f.readlines():
            for word in sent.strip().split(' '):
                frame_word_list.append(word)
    num_added_toks = tokenizer.add_tokens(frame_word_list)
    if num_added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))


def create_noisy_frames(frames, vocab="frames_list.txt", drop_prob=0.15, reorder_prob=0.1, noise_prob=0.5):
    vocab_words = []
    with open(vocab, encoding='utf-8') as f:
        for sent in f.readlines():
            for word in sent.strip().split(' '):
                vocab_words.append(word)

    frames_tokens = frames.split()
    new_order = []
    for i, token in enumerate(frames_tokens):
        if random.random() < drop_prob:
            pass
        else:
            new_order.append(token)

    num_extrawords = math.floor(len(new_order) * noise_prob)
    for e in range(num_extrawords):
        position_to_insert = random.randint(0, len(new_order))
        new_order.insert(position_to_insert, random.choice(vocab_words))

    if len(frames_tokens) > 3 and random.random() < reorder_prob:
        random.shuffle(new_order)

    return ' '.join(new_order)


def build_input_from_segments(history, reply, utterance, tokenizer, frames=None, lm_labels=False,\
                              with_eos=True, frames_type ="response_frames", generate_frames=False,
                              frames_generated='', is_test=False):
    """ Build a sequence of input from input """
    def tokenize(obj):
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)
    if not is_test:
        frames = create_noisy_frames(frames)
    frame_text = '<bof> ' + frames#utterance.get(frames_type, [''])[-1]
    frame_text += frames_generated

    bos, eos, speaker1, speaker2, bof, pad, bor = (SPECIAL_TOKENS)

    if not generate_frames:
        sequence = [bos + ' '] + history + [frame_text] + [reply + str(' '+ eos if with_eos else '')]
        sequence_temp = [sequence[0]] + [speaker1 + ' ' + s if (len(sequence) - i) % 2 else speaker2 + ' ' + s for i, s
                                         in enumerate(sequence[1:-1])]
        sequence = sequence_temp + ['<bor> ' + sequence[-1]]
    else:
        sequence = [bos + ' '] + history + [frame_text] + ['']
        sequence_temp = [sequence[0]] + [speaker1 + ' ' + s if (len(sequence) - i) % 2 else speaker2 + ' ' + s for i, s
                                         in enumerate(sequence[1:-1])]
        sequence = sequence_temp

    # print(sequence, is_test)
    # import pdb;pdb.set_trace()
    sequence = tokenize(sequence)
    # print(sequence)
    bos, eos, speaker1, speaker2, bof, pad, bor = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    # instance["token_type_ids"] = [1 if i % 2 else 0 for i, s in enumerate(sequence[:-1]) for _ in s] + [0 for i, s in enumerate(sequence[-1])]
    instance["token_type_ids"] = [1 if (len(sequence) - i) % 2 else 0 for i, s in enumerate(sequence[:-1]) for _ in s] + [0 for i, s in enumerate(sequence[-1])]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-100] * len(instance["input_ids"])
    if lm_labels:
        #original
        # instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
        # train frames as well
        bof_token_id = bof
        bof_index = sequence[-2].index(bof_token_id)
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-2])) + sequence[-2] + sequence[-1]
        # instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-2])) + [-100] * bof_index + (sequence[-2][bof_index:]) + sequence[-1]
        if is_test:
            instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + sequence[-1]

    return instance



def build_context_from_segments(history, reply, utterance, tokenizer, frames=None, lm_labels=False, with_eos=True, frames_type ="response_frames", frames_generated = ''):
    def tokenize(obj):
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)

    frame_text = '<bof> ' + frames#utterance.get(frames_type, [''])[-1]
    if frames_generated !='':
        frame_text += ' -- ' + frames_generated

    bos, eos, speaker1, speaker2, bof, pad, bor = (SPECIAL_TOKENS)
    sequence = [bos + ' '] + history + [frame_text]  + [reply + str(' ' + eos if with_eos else '')]
    # if len(frame_text) > 0 and len(history) > 0:
    #     sequence[-2] = sequence[-2] + frame_text
    sequence = [sequence[0]] + [speaker1 + ' ' + s if (len(sequence) - i) % 2 else speaker2 + ' ' + s for i, s in
                                enumerate(sequence[1:-1])]
    sequence = sequence + ['<bor> ']

    return ' '.join(sequence)

def pad_and_tensorize(batch_dict, padding):
    #https://github.com/sshleifer/transfer-learning-conv-ai/blob/batch-padding/train.py
    """ Pad the batch_dict."""
    tensors = []
    for name in MODEL_INPUTS:
        if name not in PADDED_INPUTS:
            tensors.append(torch.tensor(batch_dict[name]))
            continue
        entry = batch_dict[name]
        pad_id = padding if name != "lm_labels" else -100
        padded = pad_sequence([torch.tensor(seq) for x in entry for seq in x], batch_first=True,
                              padding_value=pad_id)
        bs, n_candidates = len(entry), len(entry[0])
        tensors.append(padded.view(bs, n_candidates, -1))
    return tensors

class ChatDataset(torch.utils.data.Dataset):

    def __init__(self, fields, pad_id):
        self.fields = fields
        self.pad_id = pad_id
        # print([(len(self.fields[f]),f) for f in MODEL_INPUTS])
    def __getitem__(self, item) -> dict:
        # print(item)
        return {f: self.fields[f][item] for f in MODEL_INPUTS}

    def collate_fn(self, examples):
        batch_dict = defaultdict(list)
        for input_name in MODEL_INPUTS:
            for e in examples:
                batch_dict[input_name].append(e[input_name])
        tensors = pad_and_tensorize(batch_dict, padding=self.pad_id)
        return tensors

    def __len__(self):
        return len(self.fields['input_ids'])

def make_data_lists(args, data_dict, tokenizer, test_only = False):
    cached_features_file = 'train_robust_features_'+ args.dataset_cache + '_' + type(tokenizer).__name__
    if args.read_from_cache and os.path.isfile(cached_features_file):
        datasets = torch.load(cached_features_file)
        return datasets

    datasets = {"train": defaultdict(list), "valid": defaultdict(list), "test": defaultdict(list)}
    for dataset_name, dataset in data_dict.items():
        if test_only and dataset_name=='train':
            continue
        if not test_only and dataset_name =='test':
            continue
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if args.num_candidates > 0 and dataset_name == 'train':
            num_candidates = min(args.num_candidates, num_candidates)
        print('num data points ', len(dataset))
        for ind, dialog in enumerate(dataset):
            if (ind+1) % 10000 == 0:
                print(ind, ' done')
                break
            for _ in range(args.data_permutations):
                for index, utterance in enumerate(dialog["utterances"]):
                    candidate_instances = defaultdict(list)
                    history = utterance["history"][-(2 * args.max_history + 1):]
                    for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                        lm_labels = bool(j == num_candidates-1)
                        frames = utterance["response_frames"][-num_candidates:][j]
                        instance = build_input_from_segments(history, candidate, utterance, tokenizer, frames=frames, lm_labels=lm_labels, is_test=(dataset_name!='train'))
                        for input_name, input_array in instance.items():
                            candidate_instances[input_name].append(input_array)
                    for k in candidate_instances.keys():
                        datasets[dataset_name][k].append(candidate_instances[k])
                    datasets[dataset_name]["mc_labels"].append(num_candidates - 1)
                    datasets[dataset_name]["n_candidates"] = num_candidates
            
    # if len(datasets['valid'] )==0:
    #     for key in datasets['train'].keys():
    #         if type(datasets['train'][key])==list:
    #             datasets['valid'][key] = datasets['train'][key][:100]
    #         else:
    #             datasets['valid'][key] = datasets['train'][key]

    torch.save(datasets, cached_features_file)

    return datasets


def get_data_loaders(args, tokenizer):
    """ Prepare the dataset for training and evaluation """
    alldata = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)

    logger.info("Build inputs and labels")
    datasets = make_data_lists(args, alldata, tokenizer)
    pad_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1])
    train_dataset = ChatDataset(datasets['train'], pad_id)
    valid_dataset = ChatDataset(datasets['valid'], pad_id)

    logger.info("Build train and validation dataloaders")
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed),
                              collate_fn=train_dataset.collate_fn)
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False,
                              collate_fn=valid_dataset.collate_fn)
    return train_loader, valid_loader, train_sampler, valid_sampler

def train():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default="gpt2", help="Path, url or short name of the model")
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=5, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
    parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--data_permutations", type=int, default=1, help="Number of permutations of permutation sentences")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--read_from_cache", action='store_true', help="If you have stored the file")
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", args.local_rank)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    tokenizer_class = GPT2Tokenizer #if "gpt2" in args.model_checkpoint else OpenAIGPTTokenizer # cant use Autotokenizer because checkpoint could be a Path
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)


    model_class = GPT2DoubleHeadsModel# if "gpt2" in args.model_checkpoint else OpenAIGPTDoubleHeadsModel
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    # Add special tokens if they are not already added
    add_special_tokens_(model, tokenizer)
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler = get_data_loaders(args, tokenizer)

    # Training function and trainer
    def update(engine, batch):
        model.train()
        batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
        input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
        (lm_loss), (mc_loss), *_ = model(
            input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
            mc_labels=mc_labels, lm_labels=lm_labels
        )
        loss = (lm_loss * args.lm_coef + mc_loss * args.mc_coef) / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
            # logger.info(tokenizer.decode(input_ids[0, -1, :].tolist()))
            # if we dont send labels to model, it doesnt return losses
            lm_logits, mc_logits, *_ = model(
                input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
            )
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return (lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, mc_labels)
    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # Linearly decrease the learning rate from lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=-100), output_transform=lambda x: (x[0][0], x[1][0])),
               "accuracy": Accuracy(output_transform=lambda x: (x[0][1], x[1][1]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args),
                    "average_accuracy": MetricsLambda(average_distributed_scalar, metrics["accuracy"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints and save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names=["loss"])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))

        log_dir = make_logdir(args.model_checkpoint)
        tb_logger = TensorboardLogger(log_dir)

        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]), event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()), another_engine=trainer), event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(log_dir, 'checkpoint', save_interval=1, n_saved=None)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})  # "getattr" takes care of distributed encapsulation

        torch.save(args, log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME))
        tokenizer.save_pretrained(log_dir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint (for easy re-loading with OpenAIGPTModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(os.path.join(log_dir, checkpoint_handler._saved[-1][1]), os.path.join(log_dir, WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()

if __name__ == "__main__":
    train()

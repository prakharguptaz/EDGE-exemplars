# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#https://www.machinelearningplus.com/python/parallel-processing-python/
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from collections import defaultdict
from pprint import pformat
import warnings
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2DoubleHeadsModel
from train_robust import SPECIAL_TOKENS, build_input_from_segments, build_context_from_segments, add_special_tokens_, get_data_loaders, ChatDataset
from utils import get_dataset
from torch.multiprocessing import Pool, Process, set_start_method

logger = logging.getLogger(__file__)


def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(history, utterance, tokenizer, model, args, frames=None, current_output=None, generate_frames=False, frames_type ="ret_response_frames"):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []
    # import pdb;pdb.set_trace()
    bor_encountered = False
    bor_number = tokenizer.convert_tokens_to_ids('<bor>')
    frames_generated = ''
    for i in range(args.max_length):
        # import pdb;pdb.set_trace()
        instance = build_input_from_segments(history, tokenizer.decode(current_output), utterance, tokenizer, frames=frames, with_eos=False, frames_generated=frames_generated, is_test=True)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            tries = 0
            while prev.item() in special_tokens_ids and tries< args.max_length:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)
                tries+=1
        if prev.item() == tokenizer.eos_token_id:#in special_tokens_ids:
            break

        if prev.item()!=bor_number:
                current_output.append(prev.item())

    return tokenizer.decode(current_output, skip_special_tokens=True)


def map_to_sample(i, history, utterance, tokenizer, model, args, frames, current_output, frames_type):
    # i, history, utterance, tokenizer, model, args, frames, current_output, frames_type  = arg_list
    text = sample_sequence(history, utterance, tokenizer, model, args, frames=frames, current_output=current_output,
                    frames_type=frames_type)
    if i%100==0:
        print(i)

    return i, text

def sample_sequence_old(history, utterance, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []
    for i in range(args.max_length):
        # import pdb;pdb.set_trace()
        instance = build_input_from_segments(history, tokenizer.decode(current_output, skip_special_tokens=True), utterance, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output


def make_data_lists(args, data_dict, tokenizer, test_only=True):
    datasets = {"test": defaultdict(list), "train": defaultdict(list), "valid": defaultdict(list)}
    data_points = []
    for dataset_name, dataset in data_dict.items():
        if dataset_name == 'valid' or dataset_name == 'train':
            continue
        num_candidates = len(dataset[0]["utterances"][0]["candidates"])
        if args.num_candidates > 0 and dataset_name == 'train':
            num_candidates = min(args.num_candidates, num_candidates)
        for ind, dialog in enumerate(dataset):
            if (ind+1) % 1000 == 0:
                print(ind)
            # if ind ==200:
            #     break
            for index, utterance in enumerate(dialog["utterances"]):
                candidate_instances = defaultdict(list)
                history = utterance["history"][-(2 * args.max_history + 1):]
                for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                    lm_labels = bool(j == num_candidates - 1)
                    if j != num_candidates - 1:
                        continue
                    # frames = utterance["response_frames"][-num_candidates:][j]
                    frames = dialog["retrieved_response_frames"]
                    # instance = build_input_from_segments(history, '', utterance, tokenizer, lm_labels=False, with_eos=False)
                    data_points.append((history, utterance, frames, tokenizer))

    return data_points

def evaluate(args, model, tokenizer, dataset, prefix="", frames_type ="response_frames"):
    eval_output_dir = args.output_dir

    dat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
    logger.info("Build inputs and labels")
    data_points = make_data_lists(args, dat, tokenizer)
    pad_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-1])

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    # logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.valid_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    if not os.path.exists(os.path.join(eval_output_dir, prefix)):
        os.makedirs(os.path.join(eval_output_dir, prefix))

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_outputs.txt")
    out_file = open(output_eval_file, "w")
    input_output_eval_file = os.path.join(eval_output_dir, prefix, "eval_input_outputs.txt")
    inout_file = open(input_output_eval_file, "w")
    model.share_memory()

    processes = []
    n_train_processes = 3

    to_process = []
    # for data_point in tqdm(data_points, desc="Evaluating"):
    # data_points = data_points[:100]
    for i, data_point in enumerate(data_points):
        history, utterance, frames, tokenizer = data_point
        to_process.append([i,history, utterance, tokenizer, model, args, frames, None, frames_type])


    pool = Pool(2)
    # results = [pool.apply(map_to_sample, args=(a)) for a in to_process]
    result_objects = [pool.apply_async(map_to_sample, args=(a)) for a in to_process]
    results = [r.get() for r in result_objects]
    results.sort(key=lambda x: x[0])

    for i, (result, data_point) in enumerate(zip(results, data_points)):
        ind, text = result
        history, utterance, frames, tokenizer = data_point
        to_process.append([i,history, utterance, tokenizer, model, args, frames, None, frames_type])
        # text = sample_sequence(history, utterance, tokenizer, model, args, frames=frames, current_output=None, frames_type=frames_type)
        frames_generated = utterance.get('frames_generated', '')
        context = build_context_from_segments(history, '', utterance, tokenizer, frames=frames, lm_labels=False, with_eos=False, frames_type=frames_type, frames_generated=frames_generated)
        out_file.write(text + '\n')
        inout_file.write(context + text + '\n')

    pool.close()
    pool.join()
    out_file.close()
    inout_file.close()
    exit()
    return True


def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="",
                        help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./test_dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="gpt2", help="Model type (openai-gpt or gpt2)",
                        choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=2, help="Batch size for validation")
    parser.add_argument("--num_candidates", type=int, default=2, help="Number of candidates for training")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")

    args = parser.parse_args()
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.output_dir = args.model_checkpoint

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_checkpoint == "":
        if args.model == 'gpt2':
            raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
        else:
            args.model_checkpoint = download_pretrained_model()

    if args.seed != 0:
        random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    logger.info("Get pretrained model and tokenizer")
    tokenizer_class, model_class = (GPT2Tokenizer, GPT2DoubleHeadsModel) if args.model == 'gpt2' else (
    OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    add_special_tokens_(model, tokenizer)

    dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)

    evaluate(args, model, tokenizer, dataset, prefix="")


if __name__ == "__main__":
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    run()

#!/usr/bin/env python3
# coding=utf-8
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
Modified from hugging face example code.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from tqdm import trange
from collections import Counter

import torch
import torch.nn.functional as F
import numpy as np

from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer

import json
import collections
import nltk
from nltk.corpus import stopwords
from difflib import SequenceMatcher
from protoqa_evaluator import *
from protoqa_evaluator.evaluation import *
from functools import partial
from pathlib import Path
import numpy as np



logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig)), ())

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
}


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def load_data_from_jsonl(data_path: Union[Path, str]) -> Dict:
    """
    Load jsonl input data, converting the answers-cleaned item to use frozenset keys.
    :param data_path: path to jsonl data
    :return: question_id indexed Dict
    """
    question_data = dict()
    with open(data_path) as data:
        for q in data:
            q_json = json.loads(q)
            if isinstance(q_json["answers-cleaned"], list):
                q_json["answers-cleaned"] = {
                    frozenset(ans_cluster["answers"]): ans_cluster["count"]
                    for ans_cluster in q_json["answers-cleaned"]
                }
            # The following attempts to handle various formats of the data
            if "normalized-question" not in q_json:
                q_json["normalized-question"] = q_json["question"][
                    "normalized-question"
                ]
            if "metadata" in q_json:
                question_data[q_json["metadata"]["id"]] = q_json
            else:
                question_data[q_json["questionid"]] = q_json
    return question_data


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0,
                    is_xlnet=False, is_xlm_mlm=False, xlm_mask_token=None, xlm_lang=None, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in trange(length):

            inputs = {'input_ids': generated}
            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    return generated

def transform_question(question):
    question = question.lower()
    question = question.replace('.', '')
    question = question.replace(':', '')
    question = question.replace('?', '')
    question = question.replace('someone', 'one person')
    question = question.replace('someplace', 'one place')
    if 'name something' in question:
        question = question.replace('name something', 'one thing')
        question += ' is'
    elif 'tell me something' in question:
        question = question.replace('tell me something', 'one thing')
        question += ' is'
    elif 'name a ' in question:
        question = question.replace('name a ', 'one ')
        question += ' is'
    elif 'name an ' in question:
        question = question.replace('name an ', 'one ')
        question += ' is'
    elif 'name' in question:
        question = question.replace('name', '')
        question += ' is'
    elif question.startswith('tell me a '):
        question = question.replace('tell me a ', 'one ')
        question += ' is'
    elif question.startswith('tell me an '):
        question = question.replace('tell me an ', 'one ')
        question += ' is'
    elif question.startswith('what '):
        question = question.replace('what', 'one')
        question += ' is'
    elif question.startswith('give me a '):
        question = question.replace('give me a ', 'one ')
        question += ' is'
    elif question.startswith('tell me '):
        question = question.replace('tell me ', '')
        question += ' is'
    elif 'which' in question:
        question = question.replace('which', 'one')
        question += ' is'
    elif 'what' in question:
        question = question.replace('what', 'one')
        question += ' is'
    elif 'how can you tell' in question:
        question = question.replace('how can you tell', 'one way to tell')
        question += ' is'
    else:
        question = 'Q: '+question +'? A: '
    return question

def get_question(data_dict):
    qidx = []
    questions = []
    for q in data_dict:
        question = data_dict[q]['normalized-question']
        transformed_question = transform_question(question)
        questions.append(transformed_question)
        qidx.append(q)
    return qidx, questions


def read_json(filename):
    with open(filename, 'r') as f:
        loaded_json = json.load(f)
    return loaded_json

def read_eval_file(filename):
    question = {}
    with open(filename) as inputfile:
        lines = inputfile.readlines()
        for line in lines:
            line = line.strip()
            part = line.split('\t')
            if len(part)>1:
                if part[0] in question:
                    ans = part[1].replace('.', '')
                    ans = ans.strip()
                    question[part[0]].append(ans)
                else:
                    ans = part[1].replace('.', '')
                    ans = ans.strip()
                    question[part[0]] = [ans]
            # no prediction
            else:
                if part[0] in question:
                    question[part[0]].append('')
                else:
                    question[part[0]] = ['']
    print('number of questions', len(question))
    return question

def match_question(true_data_withid, true_data, pred_data):
    true_result = collections.defaultdict(dict)
    pred_result = collections.defaultdict(list)
    i=0
    for q in true_data_withid:
        question = true_data_withid[q]['question']
        ans_with_count = true_data_withid[q]['answers']
        ans_set = set(ans_with_count.keys())
        question = question.lower()
        for q1 in true_data:
            ans1_set = set(true_data[q1])
            if ans_set == ans1_set:
                true_result[q] = {'question':question,
                                  'answers-cleaned':ans_with_count,
                                  'questionid': q}
                print(q, q1)
                pred_result[q] = pred_data[q1]

    assert len(true_result) == len(pred_result)
    # print(len(true_result))
    return true_result, pred_result


        # print(question)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--xlm_lang", type=str, default="", help="Optional language when used with the XLM model.")
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.69,
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--stop_token', type=str, default=".",
                        help="Token at which text generation is stopped")
    parser.add_argument('--input_file', type=str, default="./all_170.jsonl",
                        help="input file containing sentences")
    parser.add_argument('--output', type=str, default="./",
                        help="input file containing sentences")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)
    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir = './pre_trained_model')
    model = model_class.from_pretrained(args.model_name_or_path, cache_dir = './pre_trained_model')
    model.to(args.device)
    model.eval()

    en_stopwords = set(stopwords.words('english'))
    if args.length < 0 and model.config.max_position_embeddings > 0:
        args.length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < args.length:
        args.length = model.config.max_position_embeddings  # No generation bigger than model size
    elif args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    logger.info(args)
    input_filename = args.input_file
    true_dev = load_data_from_jsonl(input_filename)
    qidx, questions = get_question(true_dev)
    prediced_dev = collections.defaultdict(list)
    result = []
    i=0
    num = len(questions)
    for single_question_idx in range(len(questions)):
        print(i,'th example')
        raw_text = questions[single_question_idx]
        i+=1
        context_tokens = tokenizer.encode(raw_text, add_special_tokens=False)
        out = sample_sequence(
            model=model,
            context=context_tokens,
            num_samples=args.num_samples,
            length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            is_xlnet=False,
            is_xlm_mlm=False,
            xlm_mask_token=None,
            xlm_lang=None,
            device=args.device,
        )
        out = out[:, len(context_tokens):].tolist()
        for o in out:
            text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
            text = text[: text.find(args.stop_token)+1 if args.stop_token else None]
            text = text.strip()
            if text.endswith('.'):
                text = text[:-1]
            # print(text)
            nostop_text_list = [tok for tok in text.split(' ') if tok not in en_stopwords]
            nostop_text = " ".join(nostop_text_list)
            # print(nostop_text)
            if qidx[single_question_idx] not in prediced_dev:
                prediced_dev[qidx[single_question_idx]] = [nostop_text]
            else:
                prediced_dev[qidx[single_question_idx]].append(nostop_text)
            result.append((raw_text, nostop_text))


    ranked_predicted_dev = collections.defaultdict(list)
    sampled_answers = collections.defaultdict(list)
    for q in prediced_dev:
        counted_value = Counter(prediced_dev[q])
        sampled_answers[q] = counted_value
        ranked_list = [pair[0] for pair in counted_value.most_common(10)]
        ranked_predicted_dev[q] = ranked_list


    with open(args.output+'ranked_list.jsonl', 'w') as f:
        for key in ranked_predicted_dev:
            json.dump({key:ranked_predicted_dev[key]}, f)
            f.write('\n')
    with open(args.output+'sample_answers.json', 'w') as f:
        json.dump(sampled_answers, f)

    return None


if __name__ == '__main__':
    main()


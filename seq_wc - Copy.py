from __future__ import print_function
import datetime
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs
from model.crf import *
from model.lm_lstm_crf import *
import model.utils as utils
from model.predictor import predict_wc

import argparse
import json
import os
import sys
from tqdm import tqdm
import itertools
import functools

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluating LM-BLSTM-CRF')
    parser.add_argument('--load_arg', default='D:/PythoProjects/Datasets/checkpoint/ner_tr_cwlm_lstm_crf.json', help='path to arg json')
    parser.add_argument('--load_check_point', default='D:/PythoProjects/Datasets/checkpoint/ner_tr_cwlm_lstm_crf.model', help='path to model checkpoint file')
    parser.add_argument('--gpu',type=int, default=0, help='gpu id')
    parser.add_argument('--decode_type', choices=['label', 'string'], default='string', help='type of decode function, set `label` to couple label with text, or set `string` to insert label into test')
    parser.add_argument('--batch_size', type=int, default=50, help='size of batch')
    #parser.add_argument('--input_file', default='D:/PythoProjects/Datasets/conll003/conll003-englishversion/test.txt', help='path to input un-annotated corpus')    
    parser.add_argument('--input_file', default='D:/PythoProjects/Datasets/TezDatasets/NERResources_tobe_Distributed/WFS7with[p5].txt', help='path to input un-annotated corpus')
    parser.add_argument('--output_file', default='output_tr.txt', help='path to output file')
    args = parser.parse_args()

    print('loading dictionary')
    with open(args.load_arg, 'r') as f:
        jd = json.load(f)
    jd = jd['args']

    checkpoint_file = torch.load(args.load_check_point, map_location=lambda storage, loc: storage)
    f_map = checkpoint_file['f_map']
    l_map = checkpoint_file['l_map']
    c_map = checkpoint_file['c_map']
    in_doc_words = checkpoint_file['in_doc_words']
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    # loading corpus
    print('loading corpus')
    with codecs.open(args.input_file, 'r', 'utf-8') as f:
        lines = f.readlines()

    # converting format
    features = utils.read_features(lines)

    # build model
    print('loading model')
    ner_model = LM_LSTM_CRF(len(l_map), len(c_map), jd['char_dim'], jd['char_hidden'], jd['char_layers'], jd['word_dim'], jd['word_hidden'], jd['word_layers'], len(f_map), jd['drop_out'], large_CRF=jd['small_crf'], if_highway=jd['high_way'], in_doc_words=in_doc_words, highway_layers = jd['highway_layers'])

    ner_model.load_state_dict(checkpoint_file['state_dict'])

    if args.gpu >= 0:
        if_cuda = True
        torch.cuda.set_device(args.gpu)
        ner_model.cuda()
    else:
        if_cuda = False

    decode_label = (args.decode_type == 'label')
    predictor = predict_wc(if_cuda, f_map, c_map, l_map, f_map['<eof>'], c_map['\n'], l_map['<pad>'], l_map['<start>'], decode_label, args.batch_size, jd['caseless'])

    print('annotating')
    with open(args.output_file, 'w', encoding='utf-8') as fout:
        predictor.output_batch(ner_model, features, fout)
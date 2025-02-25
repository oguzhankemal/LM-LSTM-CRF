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
from model.evaluator import eval_wc

from torch.utils.tensorboard import SummaryWriter

import argparse
import json
import os
import sys
from tqdm import tqdm
import itertools
import functools

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def start_logging(logdir, experiment_name):
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logfilename = os.path.join(logdir, experiment_name + strftime("%Y_%m_%d,%H_%M_%S", gmtime()) + ".log")
    logginglevel = logging.DEBUG
    logging.basicConfig(level=logginglevel, filename = logfilename, format='%(asctime)s %(levelname)s %(message)s')

def decide_log_dir(self, args):
    if "exp_dir" not in args or args.exp_dir is None:
        args.exp_dir = os.path.expanduser("D:/PythoProjects/Logs/log_dir/")

    if "name" not in args or args.name is None:
        args.name = "NONAME"

    args.exp_number = self.decide_experiment_number(args.exp_dir, args.name)

    log_dir = os.path.join(args.exp_dir, "results", args.name + str(args.exp_number))

    return log_dir

def decide_experiment_number(self, exp_dir, exp_name):
    path = os.path.join(exp_dir, "results", exp_name)
    path = os.path.expanduser(path)
    numlist = [int(name.replace(path, "")[:-1]) for name in glob.glob(os.path.join(path + "*",""))]
        
    if not numlist:
        return 1
    else:
        return max(numlist) + 1

def save_args(self):
    file_name = os.path.join(self.args.log_dir, "config.json")
    with open(file_name, 'w') as outf:
        json.dump(self.args, outf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning with LM-LSTM-CRF together with Language Model')
    parser.add_argument('--rand_embedding', action='store_true', help='random initialize word embedding')
    parser.add_argument('--emb_file', default='D:/PythoProjects/Datasets/glove/glove.6B.100d.txt', help='path to pre-trained embedding')
    parser.add_argument('--train_file', default='D:/PythoProjects/Datasets/conll003/conll003-englishversion/train.txt', help='path to training file')
    parser.add_argument('--dev_file', default='D:/PythoProjects/Datasets/conll003/conll003-englishversion/valid.txt', help='path to development file')
    parser.add_argument('--test_file', default='D:/PythoProjects/Datasets/conll003/conll003-englishversion/test.txt', help='path to test file')
    
    #Turkish
    #parser.add_argument('--train_file_target', default='D:/PythoProjects/Datasets/TezDatasets/NERResources_tobe_Distributed/Train7.txt', help='path to training file')
    #parser.add_argument('--dev_file_target', default='D:/PythoProjects/Datasets/TezDatasets/NERResources_tobe_Distributed/Twitter50K.txt', help='path to development file')
    #parser.add_argument('--test_file_target', default='D:/PythoProjects/Datasets/TezDatasets/NERResources_tobe_Distributed/WFS7with[p5].txt', help='path to test file')

    #Spanish
    parser.add_argument('--train_file_target', default='D:/PythoProjects/Datasets/conll2002/esp.train', help='path to training file')
    parser.add_argument('--dev_file_target', default='D:/PythoProjects/Datasets/conll2002/esp.testb', help='path to development file')
    parser.add_argument('--test_file_target', default='D:/PythoProjects/Datasets/conll2002/esp.testa', help='path to test file')
    parser.add_argument('--emb_file_target', default='D:/PythoProjects/Datasets/glove/glove.6B.100d.txt', help='path to pre-trained embedding')

    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--batch_size', type=int, default=10, help='batch_size')
    parser.add_argument('--unk', default='unk', help='unknow-token in pre-trained embedding')
    parser.add_argument('--char_hidden', type=int, default=300, help='dimension of char-level layers')
    parser.add_argument('--word_hidden', type=int, default=300, help='dimension of word-level layers')
    parser.add_argument('--drop_out', type=float, default=0.55, help='dropout ratio')
    parser.add_argument('--epoch', type=int, default=100, help='maximum epoch number')
    parser.add_argument('--epoch_target', type=int, default=100, help='maximum epoch number')
    parser.add_argument('--start_epoch', type=int, default=0, help='start point of epoch')
    parser.add_argument('--checkpoint', default='D:/PythoProjects/SpanishCheckPoints/DS_1_To_DS_5_Shared_Char_Embedding_Without_Rand_Init_100_Epoch/DS_1_To_DS_5_Shared_Char_Embedding_Without_Rand_Init_100_Epoch_Cwlm_Lstm_Crf_', help='checkpoint path')
    parser.add_argument('--caseless', action='store_true', help='caseless or not')
    parser.add_argument('--char_dim', type=int, default=30, help='dimension of char embedding')
    parser.add_argument('--word_dim', type=int, default=100, help='dimension of word embedding')
    parser.add_argument('--char_layers', type=int, default=1, help='number of char level layers')
    parser.add_argument('--word_layers', type=int, default=1, help='number of word level layers')
    parser.add_argument('--lr', type=float, default=0.015, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.05, help='decay ratio of learning rate')
    parser.add_argument('--fine_tune', action='store_false', help='fine tune the diction of word embedding or not')
    parser.add_argument('--load_check_point', default='', help='path previous checkpoint that want to be loaded')
    parser.add_argument('--load_check_point_target', default='', help='path previous target checkpoint that want to be loaded')
    #parser.add_argument('--load_check_point', default='D:/PythoProjects/Datasets/checkpoint/ner_en_tr_cwlm_lstm_crf.model', help='path previous checkpoint that want to be loaded')
    parser.add_argument('--load_opt', action='store_true', help='also load optimizer from the checkpoint')
    parser.add_argument('--update', choices=['sgd', 'adam'], default='sgd', help='optimizer choice')

    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--clip_grad', type=float, default=5.0, help='clip grad at')
    parser.add_argument('--small_crf', action='store_false', help='use small crf instead of large crf, refer model.crf module for more details')
    parser.add_argument('--mini_count', type=float, default=5, help='thresholds to replace rare words with <unk>')
    parser.add_argument('--lambda0', type=float, default=1, help='lambda0')
    parser.add_argument('--co_train', action='store_true', help='cotrain language model')
    parser.add_argument('--patience', type=int, default=15, help='patience for early stop')
    parser.add_argument('--high_way', action='store_true', help='use highway layers')
    parser.add_argument('--highway_layers', type=int, default=1, help='number of highway layers')
    parser.add_argument('--eva_matrix', choices=['a', 'fa'], default='fa', help='use f1 and accuracy or accuracy alone')
    parser.add_argument('--least_iters', type=int, default=100, help='at least train how many epochs before stop')
    parser.add_argument('--shrink_embedding', action='store_true', help='shrink the embedding dictionary to corpus (open this if pre-trained embedding dictionary is too large, but disable this may yield better results on external corpus)')
    
    parser.add_argument('--log_enabled', type=int, default=1, help='save log')
    parser.add_argument('--log_dir', default='D:/PythoProjects/SpanishLogs/DS_1_To_DS_5_Shared_Char_Embedding_Without_Rand_Init_100_Epoch/', help='log path')
    parser.add_argument('--name', default='', help='log name')
    
    parser.add_argument('--tasks', nargs='+')
    args = parser.parse_args()
    argsvars = vars(parser.parse_args())
    TASKS = args.tasks
    TASKS = ['', '_target']
    #TASKS = ['_target']
    name = args.name

    if args.log_enabled > 0:
        if args.log_dir is None:
            args.log_dir = self.decide_log_dir(args)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        print('gpu Enabled:')

    print('setting:')
    print(args)
    print('TASKS', TASKS)

    c_map = dict()
    for task in TASKS:
        # load corpus
        print('loading corpus')
        
        if args.log_enabled > 0:
            writer = SummaryWriter(log_dir=args.log_dir)
        with codecs.open(argsvars['train_file' + task], 'r', 'utf-8') as f:
            lines = f.readlines()

        if argsvars['load_check_point'+task]:
            if os.path.isfile(argsvars['load_check_point'+task]):
                print("loading checkpoint: '{}'".format(argsvars['load_check_point'+task]))
                checkpoint_file = torch.load(argsvars['load_check_point'+task])
                args.start_epoch = checkpoint_file['epoch']
                f_map = checkpoint_file['f_map']
                l_map = checkpoint_file['l_map']
                c_map = checkpoint_file['c_map']
                train_features, train_labels = utils.read_corpus(lines)
            else:
                print("no checkpoint found at: '{}'".format(argsvars['load_check_point'+task]))
        else:
            print('constructing coding table')

            # converting format
            train_features, train_labels, f_map, l_map, c_map_task = utils.generate_corpus_char(lines, if_shrink_c_feature=True, c_thresholds=args.mini_count, if_shrink_w_feature=False)

            #Add task language's different chars to global char map
            for k in c_map_task:
                if k not in c_map:
                    #print(k)
                    c_map[k] = len(c_map)
    
    #shared char embedding
    char_embeds = nn.Embedding(len(c_map),  args.char_dim)
    
    models, dataset_loaders,dev_dataset_loaders,test_dataset_loaders = [], [],[], []
    for task in TASKS:
        # load corpus
        print('loading corpus')
        with codecs.open(argsvars['train_file' + task], 'r', 'utf-8') as f:
            lines = f.readlines()
        with codecs.open(argsvars['dev_file' + task], 'r', 'utf-8') as f:
            dev_lines = f.readlines()
        with codecs.open(argsvars['test_file' + task], 'r', 'utf-8') as f:
            test_lines = f.readlines()
        
        dev_features, dev_labels = utils.read_corpus(dev_lines)
        test_features, test_labels = utils.read_corpus(test_lines)
        args.start_epoch = 0
        if argsvars['load_check_point'+task]:
            if os.path.isfile(argsvars['load_check_point'+task]):
                print("loading checkpoint: '{}'".format(argsvars['load_check_point'+task]))
                checkpoint_file = torch.load(argsvars['load_check_point'+task])
                args.start_epoch = checkpoint_file['epoch']
                f_map = checkpoint_file['f_map']
                l_map = checkpoint_file['l_map']
                #c_map = checkpoint_file['c_map']
                in_doc_words = checkpoint_file['in_doc_words']
                train_features, train_labels = utils.read_corpus(lines)
            else:
                print("no checkpoint found at: '{}'".format(argsvars['load_check_point'+task]))
        else:
            print('constructing coding table')

            # converting format
            train_features, train_labels, f_map, l_map, c_map_task = utils.generate_corpus_char(lines, if_shrink_c_feature=True, c_thresholds=args.mini_count, if_shrink_w_feature=False)

            f_set = {v for v in f_map}
            f_map = utils.shrink_features(f_map, train_features, args.mini_count)

            if args.rand_embedding:
                print("embedding size: '{}'".format(len(f_map)))
                in_doc_words = len(f_map)
            else:
                dt_f_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), dev_features), f_set)
                dt_f_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), test_features), dt_f_set)
                print("feature size: '{}'".format(len(f_map)))
                print('loading embedding')
                if args.fine_tune:  # which means does not do fine-tune
                    f_map = {'<eof>': 0}
                f_map, embedding_tensor, in_doc_words = utils.load_embedding_wlm(argsvars['emb_file' + task], ' ', f_map, dt_f_set, args.caseless, args.unk, args.word_dim, shrink_to_corpus=args.shrink_embedding)
                print("embedding size: '{}'".format(len(f_map)))

            l_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), dev_labels))
            l_set = functools.reduce(lambda x, y: x | y, map(lambda t: set(t), test_labels), l_set)
            for label in l_set:
                if label not in l_map:
                    l_map[label] = len(l_map)
    
        print('constructing dataset')
        # construct dataset
        dataset, forw_corp, back_corp = utils.construct_bucket_mean_vb_wc(train_features, train_labels, l_map, c_map, f_map, args.caseless)
        dev_dataset, forw_dev, back_dev = utils.construct_bucket_mean_vb_wc(dev_features, dev_labels, l_map, c_map, f_map, args.caseless)
        test_dataset, forw_test, back_test = utils.construct_bucket_mean_vb_wc(test_features, test_labels, l_map, c_map, f_map, args.caseless)
    
        dataset_loader = [torch.utils.data.DataLoader(tup, args.batch_size, shuffle=True, drop_last=False) for tup in dataset]
        dataset_loaders.append(dataset_loader)

        dev_dataset_loader = [torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in dev_dataset]
        dev_dataset_loaders.append(dev_dataset_loader)

        test_dataset_loader = [torch.utils.data.DataLoader(tup, 50, shuffle=False, drop_last=False) for tup in test_dataset]
        test_dataset_loaders.append(test_dataset_loader)

        # build model
        print('building model')
        ner_model = LM_LSTM_CRF(len(l_map), len(c_map), args.char_dim, args.char_hidden, args.char_layers, args.word_dim, args.word_hidden, args.word_layers, len(f_map), args.drop_out,char_embeds, large_CRF=args.small_crf, if_highway=args.high_way, in_doc_words=in_doc_words, highway_layers = args.highway_layers)
        
        #print('parameters')
        #for parameter in ner_model.parameters():
            #print(parameter)
        #for name, param in ner_model.named_parameters():
            #if param.requires_grad:
                #print (name, param.data)
        if argsvars['load_check_point'+task]:
            ner_model.load_state_dict(checkpoint_file['state_dict'])
            state_dict_temp['word_embeds.weight'] = ner_model.word_embeds.weight
            ner_model.load_state_dict(state_dict_temp)
            #ner_model.load_state_dict(checkpoint_file['state_dict'])
        else:
            if not args.rand_embedding:
                ner_model.load_pretrained_word_embedding(embedding_tensor)
            #ner_model.rand_init(init_word_embedding=args.rand_embedding)

        if args.update == 'sgd':
            optimizer = optim.SGD(ner_model.parameters(), lr=args.lr, momentum=args.momentum)
        elif args.update == 'adam':
            optimizer = optim.Adam(ner_model.parameters(), lr=args.lr)

        if argsvars['load_check_point'+task] and args.load_opt:
            optimizer.load_state_dict(checkpoint_file['optimizer'])

        crit_lm = nn.CrossEntropyLoss()
        crit_ner = CRFLoss_vb(len(l_map), l_map['<start>'], l_map['<pad>'])
        
        ner_model.set_c_map(c_map)
        ner_model.set_l_map(l_map)
        ner_model.set_f_map(f_map)
        ner_model.set_crit_lm(crit_lm)
        ner_model.set_crit_ner(crit_ner)
        models.append(ner_model)

    #for i in range(len(TASKS)):
        #ner_model = models[i]
        #dataset_loader = dataset_loaders[i]
        #dev_dataset_loader = dev_dataset_loaders[i]
        #test_dataset_loader = test_dataset_loaders[i]
        #task = TASKS[i]

        if args.gpu >= 0:
            if_cuda = True
            print('device: ' + str(args.gpu))
            torch.cuda.set_device(args.gpu)
            ner_model.crit_ner.cuda()
            ner_model.crit_lm.cuda()
            ner_model.cuda()
            packer = CRFRepack_WC(len(ner_model.l_map), True)
        else:
            if_cuda = False
            packer = CRFRepack_WC(len(ner_model.l_map), False)


        tot_length = sum(map(lambda t: len(t), dataset_loader))

        best_f1 = float('-inf')
        best_acc = float('-inf')
        track_list = list()
        start_time = time.time()
        #epoch_list = range(args.start_epoch, args.start_epoch + argsvars['epoch'+task])
        epoch_list = range(args.start_epoch, argsvars['epoch'+task])

        patience_count = 0

        evaluator = eval_wc(packer, ner_model.l_map, args.eva_matrix)

        for epoch_idx, args.start_epoch in enumerate(epoch_list):

            epoch_loss = 0
            ner_model.train()
            for f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v in tqdm(itertools.chain.from_iterable(dataset_loader), mininterval=2,
                    desc=' - Tot it %d (epoch %d)' % (tot_length, args.start_epoch), leave=False, file=sys.stdout):
                f_f, f_p, b_f, b_p, w_f, tg_v, mask_v = packer.repack_vb(f_f, f_p, b_f, b_p, w_f, tg_v, mask_v, len_v)
                ner_model.zero_grad()
                scores = ner_model(f_f, f_p, b_f, b_p, w_f)
                loss = ner_model.crit_ner(scores, tg_v, mask_v)
                epoch_loss += utils.to_scalar(loss)
                if args.co_train:
                    cf_p = f_p[0:-1, :].contiguous()
                    cb_p = b_p[1:, :].contiguous()
                    cf_y = w_f[1:, :].contiguous()
                    cb_y = w_f[0:-1, :].contiguous()
                    cfs, _ = ner_model.word_pre_train_forward(f_f, cf_p)
                    loss = loss + args.lambda0 * ner_model.crit_lm(cfs, cf_y.view(-1))
                    cbs, _ = ner_model.word_pre_train_backward(b_f, cb_p)
                    loss = loss + args.lambda0 * ner_model.crit_lm(cbs, cb_y.view(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(ner_model.parameters(), args.clip_grad)
                optimizer.step()
            epoch_loss /= tot_length

            # update lr
            if args.update == 'sgd':
                utils.adjust_learning_rate(optimizer, args.lr / (1 + (args.start_epoch + 1) * args.lr_decay))

            # eval & save check_point

            if 'f' in args.eva_matrix:
                dev_result = evaluator.calc_score(ner_model, dev_dataset_loader)
                for label, (dev_f1, dev_pre, dev_rec, dev_acc, msg) in dev_result.items():
                    print('DEV : %s : dev_f1: %.4f dev_rec: %.4f dev_pre: %.4f dev_acc: %.4f | %s\n' % (label, dev_f1, dev_rec, dev_pre, dev_acc, msg))
                (dev_f1, dev_pre, dev_rec, dev_acc, msg) = dev_result['total']

                if args.log_enabled > 0:
                    #if task is not '':
                    writer.add_scalar("dev_f1" + name + task, dev_f1, args.start_epoch)
                    writer.add_scalar("dev_pre" + name + task, dev_pre, args.start_epoch)
                    writer.add_scalar("dev_rec" + name + task, dev_rec, args.start_epoch)
                    writer.add_scalar("dev_acc" + name + task, dev_acc, args.start_epoch)

                if dev_f1 > best_f1:
                    patience_count = 0
                    best_f1 = dev_f1

                    test_result = evaluator.calc_score(ner_model, test_dataset_loader)
                    for label, (test_f1, test_pre, test_rec, test_acc, msg) in test_result.items():
                        print('TEST : %s : test_f1: %.4f test_rec: %.4f test_pre: %.4f test_acc: %.4f | %s\n' % (label, test_f1, test_rec, test_pre, test_acc, msg))
                    (test_f1, test_rec, test_pre, test_acc, msg) = test_result['total']

                    track_list.append({'loss': epoch_loss, 'dev_f1': dev_f1, 'dev_acc': dev_acc, 'test_f1': test_f1,
                         'test_acc': test_acc})

                    print('(loss: %.4f, epoch: %d, dev F1 = %.4f, dev acc = %.4f, F1 on test = %.4f, acc on test= %.4f), saving...' % (epoch_loss,
                         args.start_epoch,
                         dev_f1,
                         dev_acc,
                         test_f1,
                         test_acc))

                    if args.log_enabled > 0:
                        #if task is not '':
                        writer.add_scalar("test_f1" + name + task, test_f1, args.start_epoch)
                        writer.add_scalar("test_pre" + name + task, test_pre, args.start_epoch)
                        writer.add_scalar("test_rec" + name + task, test_rec, args.start_epoch)
                        writer.add_scalar("test_acc" + name + task, test_acc, args.start_epoch)

                    try:
                        utils.save_checkpoint({
                            'epoch': args.start_epoch,
                            'state_dict': ner_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'f_map': ner_model.f_map,
                            'l_map': ner_model.l_map,
                            'c_map': ner_model.c_map,
                            'in_doc_words': in_doc_words
                        }, {'track_list': track_list,
                            'args': vars(args)
                            }, args.checkpoint + 'cwlm_lstm_crf'+task)
                    except Exception as inst:
                        print(inst)

                else:
                    patience_count += 1
                    print('(loss: %.4f, epoch: %d, dev F1 = %.4f, dev acc = %.4f)' % (epoch_loss,
                           args.start_epoch,
                           dev_f1,
                           dev_acc))
                    track_list.append({'loss': epoch_loss, 'dev_f1': dev_f1, 'dev_acc': dev_acc})

            else:

                dev_acc = evaluator.calc_score(ner_model, dev_dataset_loader)

                if dev_acc > best_acc:
                    patience_count = 0
                    best_acc = dev_acc
                
                    test_acc = evaluator.calc_score(ner_model, test_dataset_loader)

                    track_list.append({'loss': epoch_loss, 'dev_acc': dev_acc, 'test_acc': test_acc})

                    print('(loss: %.4f, epoch: %d, dev acc = %.4f, acc on test= %.4f), saving...' % (epoch_loss,
                         args.start_epoch,
                         dev_acc,
                         test_acc))
                    if args.log_enabled > 0:
                        #if task is not '':
                        writer.add_scalar("dev_acc" + name + task, dev_acc, start_epoch)
                        writer.add_scalar("test_acc" + name + task, test_acc, start_epoch)

                    try:
                        utils.save_checkpoint({
                            'epoch': args.start_epoch,
                            'state_dict': ner_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'f_map': ner_model.f_map,
                            'l_map': ner_model.l_map,
                            'c_map': ner_model.c_map,
                            'in_doc_words': in_doc_words
                        }, {'track_list': track_list,
                            'args': vars(args)
                            }, args.checkpoint + 'cwlm_lstm_crf'+task)
                    except Exception as inst:
                        print(inst)

                else:
                    patience_count += 1
                    print('(loss: %.4f, epoch: %d, dev acc = %.4f)' % (epoch_loss,
                           args.start_epoch,
                           dev_acc))
                    track_list.append({'loss': epoch_loss, 'dev_acc': dev_acc})

            print('epoch: ' + str(args.start_epoch) + '\t in ' + str(argsvars['epoch'+task]) + ' take: ' + str(time.time() - start_time) + ' s')

            if patience_count >= args.patience and args.start_epoch >= args.least_iters:
                break

        #print best
        if 'f' in args.eva_matrix:
            eprint(args.checkpoint + ' dev_f1: %.4f dev_rec: %.4f dev_pre: %.4f dev_acc: %.4f test_f1: %.4f test_rec: %.4f test_pre: %.4f test_acc: %.4f\n' % (dev_f1, dev_rec, dev_pre, dev_acc, test_f1, test_rec, test_pre, test_acc))
        else:
            eprint(args.checkpoint + ' dev_acc: %.4f test_acc: %.4f\n' % (dev_acc, test_acc))

    # printing summary
    print('setting:')
    print(args)

import os
import subprocess
import pickle
import logging
import time
import random
from datetime import timedelta

import numpy as np
from tqdm import tqdm


def init_experiment(params, logger_filename):
    """
    Initialize the experiment:
    - save parameters
    - create a logger
    """
    # save parameters
    get_saved_path(params)
    pickle.dump(params, open(os.path.join(params.dump_path, "params.pkl"), "wb"))

    # create a logger
    logger = create_logger(os.path.join(params.dump_path, logger_filename))
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info('The experiment will be stored in %s\n' % params.dump_path)

    return logger

class LogFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''

def create_logger(filepath):
    # create log formatter
    log_formatter = LogFormatter()
    
    # create file handler and set level to debug
    if filepath is not None:
        file_handler = logging.FileHandler(filepath, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()
    logger.reset_time = reset_time

    return logger

def get_saved_path(params):
    """
    create a directory to store the experiment
    """
    dump_path = "./" if params.dump_path == "" else params.dump_path
    if not os.path.isdir(dump_path):
        subprocess.Popen("mkdir -p %s" % dump_path, shell=True).wait()
    assert os.path.isdir(dump_path)

    # create experiment path if it does not exist
    exp_path = os.path.join(dump_path, params.exp_name)
    if not os.path.exists(exp_path):
        subprocess.Popen("mkdir -p %s" % exp_path, shell=True).wait()
    
    # generate id for this experiment
    if params.exp_id == "":
        chars = "0123456789"
        while True:
            exp_id = "".join(random.choice(chars) for _ in range(0, 3))
            if not os.path.isdir(os.path.join(exp_path, exp_id)):
                break
    else:
        exp_id = params.exp_id
    # update dump_path
    params.dump_path = os.path.join(exp_path, exp_id)
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()
    assert os.path.isdir(params.dump_path)

def load_embedding(vocab, emb_dim, emb_file):
    logger = logging.getLogger()
    embedding = np.zeros((vocab.n_words, emb_dim))
    logger.info("embedding: %d x %d" % (vocab.n_words, emb_dim))
    assert emb_file is not None
    with open(emb_file, "r") as ef:
        logger.info('Loading embedding file: %s' % emb_file)
        pre_trained = 0
        embedded_words = []
        for i, line in enumerate(ef):
            if i == 0: continue # first line would be "num of words and dimention"
            line = line.strip()
            sp = line.split()
            try:
                assert len(sp) == emb_dim + 1
            except:
                continue
            if sp[0] in vocab.word2index and sp[0] not in embedded_words:
                pre_trained += 1
                embedding[vocab.word2index[sp[0]]] = [float(x) for x in sp[1:]]
                embedded_words.append(sp[0])
        logger.info("Pre-train: %d / %d (%.2f)" % (pre_trained, vocab.n_words, pre_trained / vocab.n_words))
        # unembedded_words = [word for word in vocab.word2index if word not in embedded_words]
        # print(unembedded_words)

    return embedding

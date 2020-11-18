
from src.data_reader import preprocess, PAD_INDEX, SLOT_PAD_INDEX
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import logging
logger = logging.getLogger()

class DatasetTrain(data.Dataset):
    def __init__(self, en_data, trans_data, scale, scale_size):
        self.en_X = en_data["text"]
        self.en_y1 = en_data["intent"]
        self.en_y2 = en_data["slot"]

        self.trans_X = trans_data["text"]
        self.trans_y1 = trans_data["intent"]
        self.trans_y2 = trans_data["slot"]
        self.trans_len = len(self.trans_X)
        
        self.scale = scale
        self.scale_size = scale_size

    def __getitem__(self, index):
        return self.en_X[index], self.en_y1[index], self.en_y2[index], \
               self.trans_X[index % self.trans_len], self.trans_y1[index % self.trans_len], self.trans_y2[index % self.trans_len]
    
    def __len__(self):
        return int(len(self.en_X) * self.scale_size) if self.scale else len(self.en_X)

class DatasetEval(data.Dataset):
    def __init__(self, data):
        self.X = data["text"]
        self.y1 = data["intent"]
        self.y2 = data["slot"]

    def __getitem__(self, index):
        return self.X[index], self.y1[index], self.y2[index] 
    
    def __len__(self):
        return len(self.X)

class DatasetPretr(data.Dataset):
    def __init__(self, data):
        self.X = data["text"]
        self.y1 = data["intent"]
        self.y2 = data["slot"]

    def __getitem__(self, index):
        return self.X[index*2], self.y1[index*2], self.y2[index*2], \
               self.X[index*2+1], self.y1[index*2+1], self.y2[index*2+1]
    
    def __len__(self):
        return int(len(self.X) / 2)

def load_data(clean_txt):
    data = {"en": {}, "es": {}, "th": {}}
    # load English data
    preprocess(data, "en", clean_txt)
    # load Spanish data
    preprocess(data, "es", clean_txt)
    # load Thai data
    preprocess(data, "th", clean_txt)

    return data

def collate_fn_tr(data):
    en_X, en_y1, en_y2, trans_X, trans_y1, trans_y2 = zip(*data)
    # English
    en_lengths = [len(bs_x) for bs_x in en_X]
    en_maxLength = max(en_lengths)
    en_paddedSeqs_X = torch.LongTensor(len(en_X), en_maxLength).fill_(PAD_INDEX)
    en_paddedSeqs_y2 = torch.LongTensor(len(en_y2), en_maxLength).fill_(SLOT_PAD_INDEX)
    for i, (seq_X, seq_y2) in enumerate(zip(en_X, en_y2)):
        length = en_lengths[i]
        en_paddedSeqs_X[i, :length] = torch.LongTensor(seq_X)
        en_paddedSeqs_y2[i, :length] = torch.LongTensor(seq_y2)
    en_lengths = torch.LongTensor(en_lengths)
    en_y1 = torch.LongTensor(en_y1)
    
    # transfer language
    trans_lengths = [len(bs_x) for bs_x in trans_X]
    trans_maxLength = max(trans_lengths)
    trans_paddedSeqs_X = torch.LongTensor(len(trans_X), trans_maxLength).fill_(PAD_INDEX)
    trans_paddedSeqs_y2 = torch.LongTensor(len(trans_y2), trans_maxLength).fill_(SLOT_PAD_INDEX)
    for i, (seq_X, seq_y2) in enumerate(zip(trans_X, trans_y2)):
        length = trans_lengths[i]
        trans_paddedSeqs_X[i, :length] = torch.LongTensor(seq_X)
        trans_paddedSeqs_y2[i, :length] = torch.LongTensor(seq_y2)
    trans_lengths = torch.LongTensor(trans_lengths)
    trans_y1 = torch.LongTensor(trans_y1)

    return en_paddedSeqs_X, en_paddedSeqs_y2, en_lengths, en_y1, en_y2, \
           trans_paddedSeqs_X, trans_paddedSeqs_y2, trans_lengths, trans_y1, trans_y2

def collate_fn_eval(data):
    X, y1, y2 = zip(*data)
    lengths = [len(bs_x) for bs_x in X]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(PAD_INDEX)
    for i, seq in enumerate(X):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)
    lengths = torch.LongTensor(lengths)
    y1 = torch.LongTensor(y1)
    return padded_seqs, lengths, y1, y2

def get_dataloader(params):
    data = load_data(clean_txt=params.clean_txt)
    if params.zs == True:
        logger.info("Loading dataloader for zero-shot learning")
        dataset_tr = DatasetPretr(data["en"]["train"])
        dataset_val = DatasetEval(data[params.trans_lang]["eval"])
        dataset_test = DatasetEval(data[params.trans_lang]["test"])
    else:
        if params.tar_only == False:
            dataset_tr = DatasetTrain(data["en"]["train"], data[params.trans_lang]["train"], params.scale, params.scale_size)
        else:
            dataset_tr = DatasetEval(data[params.trans_lang]["train"])
        dataset_val = DatasetEval(data[params.trans_lang]["eval"])
        dataset_test = DatasetEval(data[params.trans_lang]["test"])

    dataloader_tr = DataLoader(dataset=dataset_tr, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn_tr if params.tar_only == False else collate_fn_eval)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn_eval)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn_eval)
    
    return dataloader_tr, dataloader_val, dataloader_test, data["en"]["vocab"], data[params.trans_lang]["vocab"]

def dataloader4pretr(params):
    data = load_data(clean_txt=params.clean_txt)
    dataset_pretr = DatasetPretr(data["en"]["train"])
    
    dataloader_pretr = DataLoader(dataset=dataset_pretr, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn_tr)

    return dataloader_pretr, data["en"]["vocab"]
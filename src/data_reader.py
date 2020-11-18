import os
import logging
import json
import csv
import re
import pickle
import random
from config import get_params
params = get_params()

logger = logging.getLogger()

PAD_INDEX = 0
UNK_INDEX = 1

# same as slot_set from function parse_tsv
slot_list = ['O', 'B-weather/noun', 'I-weather/noun', 'B-location', 'I-location', 'B-datetime', 'I-datetime', 'B-weather/attribute', 'I-weather/attribute', 'B-reminder/todo', 'I-reminder/todo', 'B-alarm/alarm_modifier', 'B-reminder/noun', 'B-reminder/recurring_period', 'I-reminder/recurring_period', 'B-reminder/reference', 'I-reminder/noun', 'B-reminder/reminder_modifier', 'I-reminder/reference', 'I-reminder/reminder_modifier', 'B-weather/temperatureUnit', 'I-alarm/alarm_modifier']
SLOT_PAD_INDEX = len(slot_list)

class Vocab():
    def __init__(self):
        self.word2index = {"PAD":PAD_INDEX, "UNK":UNK_INDEX}
        self.word2count = {}
        self.index2word = {PAD_INDEX: "PAD", UNK_INDEX: "UNK"}
        self.n_words = 2
        self.word_num = 0
        self.vocab_type = "data_vocab"

    def index_words(self, sentence):
        for word in sentence:
            self.word_num+=1
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.word2count[word] = 1
                self.n_words+=1
            else:
                self.word2count[word]+=1

# class LabelVocab():
#     def __init__(self):
#         self.word2index = {}
#         self.word2count = {}
#         self.index2word = {}
#         self.n_words = 0
#         self.index_words(slot_list + ["PAD"])
#         self.vocab_type = "label_vocab"
    
#     def index_words(self, sentence):
#         for word in sentence:
#             if word not in self.word2index:
#                 self.word2index[word] = self.n_words
#                 self.index2word[self.n_words] = word
#                 self.word2count[word] = 1
#                 self.n_words+=1
#             else:
#                 self.word2count[word]+=1

def get_vocab(word_set):
    vocab = Vocab()
    vocab.index_words(word_set)
    return vocab

def binarize_data(data, intent_set, slot_set, vocab, lang, isTrain):
    data_bin = {"text": [], "slot": [], "intent": []}
    # binarize intent
    for intent in data["intent"]:
        index = intent_set.index(intent)
        data_bin["intent"].append(index)
    # binarize text
    for text_tokens in data["text"]:
        text_bin = []
        for token in text_tokens:
            text_bin.append(vocab.word2index[token])
        data_bin["text"].append(text_bin)
    # binarize slot
    for slot in data["slot"]:
        slot_bin = []
        for slot_item in slot:
            index = slot_set.index(slot_item)
            slot_bin.append(index)
        data_bin["slot"].append(slot_bin)
    
    # if params.trim == False or lang == "en" or isTrain == False:
    assert len(data_bin["slot"]) == len(data_bin["text"]) == len(data_bin["intent"])
    for text, slot in zip(data_bin["text"], data_bin["slot"]):
        assert len(text) == len(slot)
    return data_bin
    # else:
    #     trim_size = int(len(data_bin["slot"]) * params.prop)
    #     logger.info("Trimming data. Reduce %s training data size to %d" % (lang, trim_size))
    #     zip_data = list(zip(data_bin["text"], data_bin["intent"], data_bin["slot"]))
    #     random.shuffle(zip_data)
    #     text_list, intent_list, slot_list = zip(*zip_data)
    #     trim_data_bin = {"text": [], "slot": [], "intent": []}
    #     cnt = 0
    #     for data_text, intent_label, slot_label in zip(text_list, intent_list, slot_list):
    #         if slot_label in trim_data_bin["slot"]: continue
    #         else:
    #             trim_data_bin["text"].append(data_text)
    #             trim_data_bin["slot"].append(slot_label)
    #             trim_data_bin["intent"].append(intent_label)
    #             cnt = cnt + 1
    #             if cnt > trim_size:
    #                 break
    #     return trim_data_bin

def parse_tsv(data_path, intent_set=[], slot_set=["O"], istrain=True):
    """
    Input: 
        data_path: the path of data
        intent_set: set of intent (empty if it is train data)
        slot_set: set of slot type (empty if it is train data)
    Output:
        data_tsv: {"text": [[token1, token2, ...], ...], "slot": [[slot_type1, slot_type2, ...], ...], "intent": [intent_type, ...]}
        intent_set: set of intent
        slot_set: set of slot type
    """
    slot_type_list = ["alarm", "datetime", "location", "reminder", "weather"]
    data_tsv = {"text": [], "slot": [], "intent": []}
    with open(data_path) as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")
        for i, line in enumerate(reader):
            intent = line[0]
            if istrain == True and intent not in intent_set: intent_set.append(intent)
            if istrain == False and intent not in intent_set:
                intent_set.append(intent)
            slot_splits = line[1].split(",")
            slot_line = []
            slot_flag = True
            if line[1] != '':
                for item in slot_splits:
                    item_splits = item.split(":")
                    assert len(item_splits) == 3
                    slot_item = {"start": item_splits[0], "end": item_splits[1], "slot": item_splits[2]}
                    flag = False
                    for slot_type in slot_type_list:
                        if slot_type in slot_item["slot"]:
                            flag = True

                    if flag == False:
                        slot_flag = False
                        break
                    slot_line.append(slot_item)
            
            if slot_flag == False:
                # slot flag not correct
                continue

            token_part = json.loads(line[4])
            tokens = token_part["tokenizations"][0]["tokens"]
            tokenSpans = token_part["tokenizations"][0]["tokenSpans"]

            data_tsv["text"].append(tokens)
            data_tsv["intent"].append(intent)
            slots = []
            for tokenspan in tokenSpans:
                nolabel = True
                for slot_item in slot_line:
                    start = tokenspan["start"]
                    # if int(start) >= int(slot_item["start"]) and int(start) < int(slot_item["end"]):
                    if int(start) == int(slot_item["start"]):
                        nolabel = False
                        slot_ = "B-" + slot_item["slot"]
                        slots.append(slot_)
                        if slot_ not in slot_set:
                            slot_set.append(slot_)
                        break
                    if int(start) > int(slot_item["start"]) and int(start) < int(slot_item["end"]):
                        nolabel = False
                        slot_ = "I-" + slot_item["slot"]
                        slots.append(slot_)
                        if slot_ not in slot_set:
                            slot_set.append(slot_)
                        break
                if nolabel == True: slots.append("O")
            data_tsv["slot"].append(slots)

            assert len(slots) == len(tokens)

    return data_tsv, intent_set, slot_set

def clean_text(data, lang):
    # detect pattern
    # detect <TIME>
    pattern_time1 = re.compile(r"[0-9]+[ap]")
    pattern_time2 = re.compile(r"[0-9]+[;.h][0-9]+")
    pattern_time3 = re.compile(r"[ap][.][am]")
    pattern_time4 = range(2000, 2020)
    # pattern_time5: token.isdigit() and len(token) == 3

    pattern_time_th1 = re.compile(r"[\u0E00-\u0E7F]+[0-9]+")
    pattern_time_th2 = re.compile(r"[0-9]+[.]*[0-9]*[\u0E00-\u0E7F]+")
    pattern_time_th3 = re.compile(r"[0-9]+[.][0-9]+")

    # detect <LAST>
    pattern_last1 = re.compile(r"[0-9]+min")
    pattern_last2 = re.compile(r"[0-9]+h")
    pattern_last3 = re.compile(r"[0-9]+sec")

    # detect <DATE>
    pattern_date1 = re.compile(r"[0-9]+st")
    pattern_date2 = re.compile(r"[0-9]+nd")
    pattern_date3 = re.compile(r"[0-9]+rd")
    pattern_date4 = re.compile(r"[0-9]+th")

    # detect <LOCATION>: token.isdigit() and len(token) == 5
    
    # detect <NUMBER>: token.isdigit()
    
    # for English: replace contain n't with not
    # for English: remove 's, 'll, 've, 'd, 'm
    remove_list = ["'s", "'ll", "'ve", "'d", "'m"]
    
    data_clean = {"text": [], "slot": [], "intent": []}
    data_clean["slot"] = data["slot"]
    data_clean["intent"] = data["intent"]
    for token_list in data["text"]:
        token_list_clean = []
        for token in token_list:
            new_token = token
            # detect <TIME>
            if lang != "th" and ( bool(re.match(pattern_time1, token)) or bool(re.match(pattern_time2, token)) or bool(re.match(pattern_time3, token)) or token in pattern_time4 or (token.isdigit() and len(token)==3) ):
                new_token = "<TIME>"
                token_list_clean.append(new_token)
                continue
            if lang == "th" and ( bool(re.match(pattern_time_th1, token)) or bool(re.match(pattern_time_th2, token)) or bool(re.match(pattern_time_th3, token)) ):
                new_token = "<TIME>"
                token_list_clean.append(new_token)
                continue
            # detect <LAST>
            if lang == "en" and ( bool(re.match(pattern_last1, token)) or bool(re.match(pattern_last2, token)) or bool(re.match(pattern_last3, token)) ):
                new_token = "<LAST>"
                token_list_clean.append(new_token)
                continue
            # detect <DATE>
            if lang == "en" and ( bool(re.match(pattern_date1, token)) or bool(re.match(pattern_date2, token)) or bool(re.match(pattern_date3, token)) or bool(re.match(pattern_date4, token)) ):
                new_token = "<DATA>"
                token_list_clean.append(new_token)
                continue
            # detect <LOCATION>
            if lang != "th" and ( token.isdigit() and len(token)==5 ):
                new_token = "<LOCATION>"
                token_list_clean.append(new_token)
                continue
            # detect <NUMBER>
            if token.isdigit():
                new_token = "<NUMBER>"
                token_list_clean.append(new_token)
                continue
            if lang == "en" and ("n't" in token):
                new_token = "not"
                token_list_clean.append(new_token)
                continue
            if lang == "en":
                for item in remove_list:
                    if item in token:
                        new_token = token.replace(item, "")
                        break
            token_list_clean.append(new_token)
        
        assert len(token_list_clean) == len(token_list)
        data_clean["text"].append(token_list_clean)
    
    return data_clean

def preprocess(data, lang, clean_txt=True):
    # preprocess from raw (lang) data
    # print("============ Preprocess %s data ============" % lang)
    logger.info("============ Preprocess %s data ============" % lang)

    data_folder = os.path.join('./data/', lang)
    if params.trim == False or lang == "en":
        logger.info("============ Loading all %s training data ============" % lang)
        train_path = os.path.join(data_folder, "train-%s.tsv" % lang)
    else:
        if params.prop == 0.01:
            logger.info("============ Loading 1 percent %s training data ============" % lang)
            train_path = os.path.join(data_folder, "train-%s.1per.tsv" % lang)
        elif params.prop == 0.03:
            logger.info("============ Loading 3 percent %s training data ============" % lang)
            train_path = os.path.join(data_folder, "train-%s.3per.tsv" % lang)
        elif params.prop == 0.05:
            logger.info("============ Loading 5 percent %s training data ============" % lang)
            train_path = os.path.join(data_folder, "train-%s.5per.tsv" % lang)
    eval_path = os.path.join(data_folder, "eval-%s.tsv" % lang)
    test_path = os.path.join(data_folder, "test-%s.tsv" % lang)

    data_train, intent_set, slot_set = parse_tsv(train_path)
    data_eval, intent_set, slot_set = parse_tsv(eval_path, intent_set=intent_set, slot_set=slot_set, istrain=False)
    data_test, intent_set, slot_set = parse_tsv(test_path, intent_set=intent_set, slot_set=slot_set, istrain=False)

    assert len(intent_set) == len(set(intent_set))
    assert len(slot_set) == len(set(slot_set))

    # logger.info("number of intent in %s is %s" % (lang, len(intent_set)))
    # logger.info("number of slot in %s is %s" % (lang, len(slot_set)))
    # print("number of intent in %s is %s" % (lang, len(intent_set)))
    # print("number of slot in %s is %s" % (lang, len(slot_set)))
    
    if clean_txt == True:
        # clean_data
        logger.info("cleaning data on %s language" % lang)
        data_train = clean_text(data_train, lang)
        data_eval = clean_text(data_eval, lang)
        data_test = clean_text(data_test, lang)

    word_set = []
    for wordlist in data_train["text"]: word_set.extend(wordlist)
    for wordlist in data_eval["text"]: word_set.extend(wordlist)
    for wordlist in data_test["text"]: word_set.extend(wordlist)
    word_set = set(word_set)
    assert len(word_set) == len(set(word_set))

    vocab = get_vocab(word_set)
    # logger.info("vocab size of %s is %d" % (lang, vocab.word_num))
    # print("vocab size of %s is %d" % (lang, vocab.word_num))

    data_train_bin = binarize_data(data_train, intent_set, slot_set, vocab, lang, isTrain=True)
    data_eval_bin = binarize_data(data_eval, intent_set, slot_set, vocab, lang, isTrain=False)
    data_test_bin = binarize_data(data_test, intent_set, slot_set, vocab, lang, isTrain=False)
    data[lang] = {"train": data_train_bin, "eval": data_eval_bin, "test": data_test_bin, "vocab": vocab}


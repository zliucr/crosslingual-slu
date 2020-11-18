
import torch
import torch.nn as nn

import os
from tqdm import tqdm

class DistributionGenerator(object):
    def __init__(self, params):
        # load sentence information
        self.sent_en = params.sent_en
        self.sent_th = params.sent_th
        self.indices_en = params.indices_en
        self.indices_th = params.indices_th
        self.n_points = params.n_points
        self.adv = params.adv
        
        # load model
        path = os.path.join(params.dump_path, "best_model.pth")
        self.slu_model = torch.load(path)
        
        # load vocabulary
        if self.adv:
            self.vocab_en = self.slu_model.model.lstm.vocab_en
            self.vocab_trans = self.slu_model.model.lstm.vocab_trans
        else:
            self.vocab_en = self.slu_model.lstm.vocab_en
            self.vocab_trans = self.slu_model.lstm.vocab_trans

    def gen_distribution(self):
        self.slu_model.eval()
        
        # Enlgish sentences
        sent_en_idx = [[ self.vocab_en.word2index[token] for token in self.sent_en.split() ]]
        sent_en_idx = torch.LongTensor(sent_en_idx)
        sent_en_idx = sent_en_idx.cuda()
        # English word indices
        indices_en_list = self.indices_en.split("-")
        indices_en_list = [ int(idx) for idx in indices_en_list ]

        # Thai
        sent_th_idx = [[ self.vocab_trans.word2index[token] for token in self.sent_th.split() ]]
        sent_th_idx = torch.LongTensor(sent_th_idx)
        sent_th_idx = sent_th_idx.cuda()
        # Thai word indices
        indices_th_list = self.indices_th.split("-")
        indices_th_list = [ int(idx) for idx in indices_th_list ]

        # generate a list of points for en and th
        if self.adv:
            lstm_layer_en = self.slu_model.model.lstm(sent_en_idx, "en")
            lstm_layer_th = self.slu_model.model.lstm(sent_th_idx, "th")
        else:
            lstm_layer_en = self.slu_model.lstm(sent_en_idx, "en")
            lstm_layer_th = self.slu_model.lstm(sent_th_idx, "th")

        word1_points_en_list, word2_points_en_list = [], []
        word1_points_th_list, word2_points_th_list = [], []
        for i in tqdm(range(0, self.n_points)):
            # Points for English
            if self.adv:
                latent_variable_sent_en = self.slu_model.model.slot_predictor.gen_latent_variables(lstm_layer_en)  # (1, seq_len, lvm_dim)
            else:
                latent_variable_sent_en = self.slu_model.slot_predictor.gen_latent_variables(lstm_layer_en)  # (1, seq_len, lvm_dim)
            word1_points_en_list.append(latent_variable_sent_en[0][indices_en_list[0]].data.cpu().numpy())
            word2_points_en_list.append(latent_variable_sent_en[0][indices_en_list[1]].data.cpu().numpy())

            # Points for Thai
            if self.adv:
                latent_variable_sent_th =self.slu_model.model.slot_predictor.gen_latent_variables(lstm_layer_th)  # (1, seq_len, lvm_dim)
            else:
                latent_variable_sent_th =self.slu_model.slot_predictor.gen_latent_variables(lstm_layer_th)  # (1, seq_len, lvm_dim)
            word1_points_th_list.append(latent_variable_sent_th[0][indices_th_list[0]].data.cpu().numpy())
            word2_points_th_list.append(latent_variable_sent_th[0][indices_th_list[1]].data.cpu().numpy())

        return word1_points_en_list, word2_points_en_list, word1_points_th_list, word2_points_th_list


from src.data_reader import slot_list as index2slot
from src.data_reader import SLOT_PAD_INDEX
import torch
import torch.nn as nn

import os
from tqdm import tqdm
import numpy as np
import logging
logger = logging.getLogger()

from sklearn.metrics import accuracy_score
from .conll2002_metrics import *

class SLUTrainer(object):
    def __init__(self, params, slu_model, pretrain_flag=False):
        self.la_reg = params.la_reg
        if self.la_reg == True:
            self.cos_dist = nn.CosineSimilarity(dim=-1, eps=1e-6)
            self.loss_fn_cos = nn.MSELoss()

        self.slu_model = slu_model

        self.lr = params.lr
        self.rmcrf = params.rmcrf
        self.pretr_la_enc = params.pretr_la_enc
        self.adversarial = params.adv
        self.intent_adv = params.intent_adv
        self.num_slot = params.num_slot
        self.num_intent = params.num_intent

        self.loss_fn = nn.CrossEntropyLoss()
        if self.adversarial == True and pretrain_flag == False:
            self.softmax = nn.Softmax(dim=-1)
            self.loss_fn_mse = nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.slu_model.model.parameters(), lr=self.lr)   # optimizer for slu model
            self.optimizer_udg = torch.optim.Adam(self.slu_model.uniform_distri_gen.parameters(), lr=self.lr)   # optimizer for uniform distribution generator(udg)
        else:
            self.optimizer = torch.optim.Adam(self.slu_model.parameters(), lr=self.lr)
        self.early_stop = params.early_stop
        self.dump_path = params.dump_path
        self.epoch_patient = params.epoch_patient
        self.no_improvement_num = 0
        self.best_intent_acc = 0
        self.best_slot_f1 = 0
        
        self.stop_training_flag = False
    
    def joint_train_step(self, e, x_en, lengths_en, x_trans, lengths_trans, y1_en, y2_en, y1_trans, y2_trans, padded_y2_en=None, padded_y2_trans=None):
        self.slu_model.train()

        if self.la_reg == False:
            if self.adversarial == True:
                if self.intent_adv == True:
                    intent_pred_en, slot_pred_en, adv_pred_en, adv_pred_intent_en, intent_pred_trans, slot_pred_trans, adv_pred_trans, adv_pred_intent_trans = self.slu_model(x_en, lengths_en, x_trans, lengths_trans)
                else:
                    intent_pred_en, slot_pred_en, adv_pred_en, intent_pred_trans, slot_pred_trans, adv_pred_trans = self.slu_model(x_en, lengths_en, x_trans, lengths_trans)
            else:
                intent_pred_en, slot_pred_en, intent_pred_trans, slot_pred_trans = self.slu_model(x_en, lengths_en, x_trans, lengths_trans)
        else:
            if self.adversarial == True:
                if self.intent_adv == True:
                    intent_pred_en, slot_pred_en, adv_pred_en, adv_pred_intent_en, intent_pred_trans, slot_pred_trans, adv_pred_trans, adv_pred_intent_trans, utter_repre_en, label_repre_en, utter_repre_trans, label_repre_trans = self.slu_model(x_en, lengths_en, x_trans, lengths_trans, padded_y2_en, padded_y2_trans)
                else:
                    intent_pred_en, slot_pred_en, adv_pred_en, intent_pred_trans, slot_pred_trans, adv_pred_trans, utter_repre_en, label_repre_en, utter_repre_trans, label_repre_trans = self.slu_model(x_en, lengths_en, x_trans, lengths_trans, padded_y2_en, padded_y2_trans)
            else:
                intent_pred_en, slot_pred_en, intent_pred_trans, slot_pred_trans, utter_repre_en, label_repre_en, utter_repre_trans, label_repre_trans = self.slu_model(x_en, lengths_en, x_trans, lengths_trans, padded_y2_en, padded_y2_trans)
            
            # update gradient for cos distance loss
            self.optimizer.zero_grad()
            label_cos_loss, utter_cos_loss = self.get_label_regularization_loss(utter_repre_en, label_repre_en, utter_repre_trans, label_repre_trans)
            if self.pretr_la_enc == False:
                label_cos_loss.backward(retain_graph=True)
            utter_cos_loss.backward(retain_graph=True)
            self.optimizer.step()
        
        # adversarial lvm training
        if self.adversarial == True:
            if self.la_reg == False:
                # generate padded_y2 labels
                padded_y2_en = self.pad_label_for_adv(lengths_en, y2_en)
                padded_y2_trans = self.pad_label_for_adv(lengths_trans, y2_trans)

            # optimize our model to let udg make correct classifications
            size_en = adv_pred_en.size()   # (bsz, seq_len, num_slot+1)
            model_en_adv_loss = self.loss_fn(adv_pred_en.view(size_en[0]*size_en[1], self.num_slot+1), padded_y2_en.view(size_en[0]*size_en[1]))
            size_trans = adv_pred_trans.size()
            model_trans_adv_loss = self.loss_fn(adv_pred_trans.view(size_trans[0]*size_trans[1], self.num_slot+1), padded_y2_trans.view(size_trans[0]*size_trans[1]))

            # adversarial for intent
            if self.intent_adv == True:
                model_en_adv_intent_loss = self.loss_fn(adv_pred_intent_en, y1_en)
                model_trans_adv_intent_loss = self.loss_fn(adv_pred_intent_trans, y1_trans)

            if e >= self.epoch_patient:
                self.optimizer.zero_grad()
                model_en_adv_loss.backward(retain_graph=True)
                model_trans_adv_loss.backward(retain_graph=True)
                self.optimizer.step()
            
            # adversarial for intent
            if self.intent_adv == True and e >= 2:
                self.optimizer.zero_grad()
                model_en_adv_intent_loss.backward(retain_graph=True)
                model_trans_adv_intent_loss.backward(retain_graph=True)
                self.optimizer.step()

            # generate uniform distribution based on bsz
            adv_pred_en = self.softmax(adv_pred_en)
            adv_pred_trans = self.softmax(adv_pred_trans)
            uniform_distri_en = torch.FloatTensor(size_en).fill_(1/(self.num_slot+1)).cuda()
            udg_en_adv_loss = self.loss_fn_mse(adv_pred_en.view(size_en[0]*size_en[1], self.num_slot+1), uniform_distri_en.view(size_en[0]*size_en[1], self.num_slot+1))

            uniform_distri_trans = torch.FloatTensor(size_trans).fill_(1/(self.num_slot+1)).cuda()
            udg_trans_adv_loss = self.loss_fn_mse(adv_pred_trans.view(size_trans[0]*size_trans[1], self.num_slot+1), uniform_distri_trans.view(size_trans[0]*size_trans[1], self.num_slot+1))

            # adversarial for intent
            if self.intent_adv == True:
                adv_pred_intent_en = self.softmax(adv_pred_intent_en)
                adv_pred_intent_trans = self.softmax(adv_pred_intent_trans)
                intent_pred_size = adv_pred_intent_en.size()
                uniform_distri_intent = torch.FloatTensor(intent_pred_size).fill_(1/(self.num_intent)).cuda()

                udg_en_adv_intent_loss = self.loss_fn_mse(adv_pred_intent_en, uniform_distri_intent)
                udg_trans_adv_intent_loss = self.loss_fn_mse(adv_pred_intent_trans, uniform_distri_intent)
                
            if e < 1:
                # optimize udg to generate uniform distribution
                self.optimizer_udg.zero_grad()
                udg_en_adv_loss.backward(retain_graph=True)
                udg_trans_adv_loss.backward(retain_graph=True)
                self.optimizer_udg.step()

            # adversarial for intent
            if self.intent_adv == True and e < 2:
                self.optimizer_udg.zero_grad()
                udg_en_adv_intent_loss.backward(retain_graph=True)
                udg_trans_adv_intent_loss.backward(retain_graph=True)
                self.optimizer_udg.step()

        # English
        self.optimizer.zero_grad()
        # intent
        intent_loss_en = self.loss_fn(intent_pred_en, y1_en)
        intent_loss_en.backward(retain_graph=True)
        # slot
        if self.rmcrf == False:
            if self.adversarial == True:
                slot_loss_en = self.slu_model.model.slot_predictor.crf_loss(slot_pred_en, lengths_en, y2_en)
            else:
                slot_loss_en = self.slu_model.slot_predictor.crf_loss(slot_pred_en, lengths_en, y2_en)
            slot_loss_en.backward()
            self.optimizer.step()
            slot_loss_en_item = slot_loss_en.item()
        else:
            slot_loss_list_en = []
            for i, length in enumerate(lengths_en):
                slot_pred_each = slot_pred_en[i][:length]
                slot_loss = self.loss_fn(slot_pred_each, torch.LongTensor(y2_en[i]).cuda())
                slot_loss.backward(retain_graph=True)
                slot_loss_list_en.append(slot_loss.item())
            self.optimizer.step()
            slot_loss_en_item = np.mean(slot_loss_list_en)
        # Transfer language
        self.optimizer.zero_grad()
        # intent
        intent_loss_trans = self.loss_fn(intent_pred_trans, y1_trans)
        intent_loss_trans.backward(retain_graph=True)
        # slot
        if self.rmcrf == False:
            if self.adversarial == True:
                slot_loss_trans = self.slu_model.model.slot_predictor.crf_loss(slot_pred_trans, lengths_trans, y2_trans)
            else:
                slot_loss_trans = self.slu_model.slot_predictor.crf_loss(slot_pred_trans, lengths_trans, y2_trans)
            slot_loss_trans.backward()
            self.optimizer.step()
            slot_loss_trans_item = slot_loss_trans.item()
        else:
            slot_loss_list_trans = []
            for i, length in enumerate(lengths_trans):
                slot_pred_each = slot_pred_trans[i][:length]
                slot_loss = self.loss_fn(slot_pred_each, torch.LongTensor(y2_trans[i]).cuda())
                slot_loss.backward(retain_graph=True)
                slot_loss_list_trans.append(slot_loss.item())
            self.optimizer.step()
            slot_loss_trans_item = np.mean(slot_loss_list_trans)

        if self.la_reg == False:
            if self.adversarial == True:
                if self.intent_adv == True:
                    return intent_loss_en.item(), slot_loss_en_item, intent_loss_trans.item(), slot_loss_trans_item, model_en_adv_loss.item(), model_trans_adv_loss.item(), udg_en_adv_loss.item(), udg_trans_adv_loss.item(), model_en_adv_intent_loss.item(), model_trans_adv_intent_loss.item(), udg_en_adv_intent_loss.item(), udg_trans_adv_intent_loss.item()
                else:
                    return intent_loss_en.item(), slot_loss_en_item, intent_loss_trans.item(), slot_loss_trans_item, model_en_adv_loss.item(), model_trans_adv_loss.item(), udg_en_adv_loss.item(), udg_trans_adv_loss.item()
            else:
                return intent_loss_en.item(), slot_loss_en_item, intent_loss_trans.item(), slot_loss_trans_item
        else:
            if self.adversarial == True:
                if self.intent_adv == True:
                    return intent_loss_en.item(), slot_loss_en_item, intent_loss_trans.item(), slot_loss_trans_item, utter_cos_loss.item(), model_en_adv_loss.item(), model_trans_adv_loss.item(), udg_en_adv_loss.item(), udg_trans_adv_loss.item(), model_en_adv_intent_loss.item(), model_trans_adv_intent_loss.item(), udg_en_adv_intent_loss.item(), udg_trans_adv_intent_loss.item()
                else:
                    return intent_loss_en.item(), slot_loss_en_item, intent_loss_trans.item(), slot_loss_trans_item, utter_cos_loss.item(), model_en_adv_loss.item(), model_trans_adv_loss.item(), udg_en_adv_loss.item(), udg_trans_adv_loss.item()
            else:
                return intent_loss_en.item(), slot_loss_en_item, intent_loss_trans.item(), slot_loss_trans_item, utter_cos_loss.item()
    
    def pad_label_for_adv(self, lengths, y):
        bsz = len(lengths)
        max_len = torch.max(lengths)
        padded_y = torch.LongTensor(bsz, max_len).fill_(SLOT_PAD_INDEX)
        for i in range(bsz):
            length = lengths[i]
            y_i = y[i]
            padded_y[i, 0:length] = torch.LongTensor(y_i)

        padded_y = padded_y.cuda()
        return padded_y
    
    def get_label_regularization_loss(self, utter_repre_en, label_repre_en, utter_repre_trans, label_repre_trans, label_loss_only=False):
        utter_cos_dist = self.cos_dist(utter_repre_en, utter_repre_trans)
        label_cos_dist = self.cos_dist(label_repre_en, label_repre_trans)
        
        utter_cos_dist_no_grad = utter_cos_dist.detach()
        label_cos_loss = self.loss_fn_cos(label_cos_dist, utter_cos_dist_no_grad)
        # utter_cos_dist.requires_grad = True
        
        if label_loss_only == False:
            label_cos_dist_no_grad = label_cos_dist.detach()
            utter_cos_loss = self.loss_fn_cos(utter_cos_dist, label_cos_dist_no_grad)
            # label_cos_dist.requires_grad = True
            return label_cos_loss, utter_cos_loss
        else:
            return label_cos_loss
    
    def single_train_step(self, X, lengths, y1, y2):
        self.slu_model.train()
        _, _, intent_pred, slot_pred = self.slu_model(None, None, X, lengths)

        self.optimizer.zero_grad()
        intent_loss = self.loss_fn(intent_pred, y1)
        intent_loss.backward(retain_graph=True)
        # slot
        slot_loss = self.slu_model.slot_predictor.crf_loss(slot_pred, lengths, y2)
        slot_loss.backward()
        self.optimizer.step()

        return intent_loss.item(), slot_loss.item()
    
    def pretr_label_encoder_step(self, e, x_1, lengths_1, x_2, lengths_2, y1_1, y2_1, y1_2, y2_2, padded_y2_1, padded_y2_2):
        self.slu_model.train()

        intent_pred_1, slot_pred_1, intent_pred_2, slot_pred_2, utter_repre_1, label_repre_1, utter_repre_2, label_repre_2 = self.slu_model(x_1, lengths_1, x_2, lengths_2, padded_y2_1, padded_y2_2)

        # update gradient for cos distance loss
        label_cos_loss = self.get_label_regularization_loss(utter_repre_1, label_repre_1, utter_repre_2, label_repre_2, label_loss_only=True)
        if e > 2:
            self.optimizer.zero_grad()
            label_cos_loss.backward()
            self.optimizer.step()

        # first sample
        self.optimizer.zero_grad()
        # intent
        intent_loss_1 = self.loss_fn(intent_pred_1, y1_1)
        intent_loss_1.backward(retain_graph=True)
        # slot
        slot_loss_1 = self.slu_model.slot_predictor.crf_loss(slot_pred_1, lengths_1, y2_1)
        slot_loss_1.backward()
        self.optimizer.step()
        
        # second sample
        self.optimizer.zero_grad()
        # intent
        intent_loss_2 = self.loss_fn(intent_pred_2, y1_2)
        intent_loss_2.backward(retain_graph=True)
        # slot
        slot_loss_2 = self.slu_model.slot_predictor.crf_loss(slot_pred_2, lengths_2, y2_2)
        slot_loss_2.backward()
        self.optimizer.step()
        
        return intent_loss_1.item(), slot_loss_1.item(), intent_loss_2.item(), slot_loss_2.item(), label_cos_loss.item()
        
    def evaluate(self, dataloader, istestset=False):
        self.slu_model.eval()

        intent_pred, slot_pred = [], []
        y1_list, y2_list = [], []
        pbar = tqdm(enumerate(dataloader),total=len(dataloader))
        for i, (X, lengths, y1, y2) in pbar:
            y1_list.append(y1.data.cpu().numpy())
            y2_list.extend(y2) # y2 is a list
            X, lengths = X.cuda(), lengths.cuda()
            _, _, intent_pred_trans, slot_pred_trans = self.slu_model(None, None, X, lengths, istestset=istestset)

            # for intent_pred
            intent_pred.append(intent_pred_trans.data.cpu().numpy())
            # for slot_pred
            if self.adversarial == True:
                slot_pred_batch = self.slu_model.model.slot_predictor.crf_decode(slot_pred_trans, lengths) if self.rmcrf == False else self.slu_model.model.slot_predictor.decode(slot_pred_trans, lengths)
            else:
                slot_pred_batch = self.slu_model.slot_predictor.crf_decode(slot_pred_trans, lengths) if self.rmcrf == False else self.slu_model.slot_predictor.decode(slot_pred_trans, lengths)
            slot_pred.extend(slot_pred_batch)

        # concatenation
        intent_pred = np.concatenate(intent_pred, axis=0)
        intent_pred = np.argmax(intent_pred, axis=1)
        slot_pred = np.concatenate(slot_pred, axis=0)
        if self.rmcrf == True:
            slot_pred = np.argmax(slot_pred, axis=1)
        y1_list = np.concatenate(y1_list, axis=0)
        y2_list = np.concatenate(y2_list, axis=0)
        intent_acc = accuracy_score(y1_list, intent_pred)

        # calcuate f1 score
        y2_list = list(y2_list)
        slot_pred = list(slot_pred)
        lines = []
        for pred_index, gold_index in zip(slot_pred, y2_list):
            pred_slot = index2slot[pred_index]
            gold_slot = index2slot[gold_index]
            lines.append("w" + " " + pred_slot + " " + gold_slot)
        results = conll2002_measure(lines)
        slot_f1 = results["fb1"]

        if istestset == False:
            if intent_acc > self.best_intent_acc:
                self.best_intent_acc = intent_acc
            if slot_f1 > self.best_slot_f1:
                self.best_slot_f1 = slot_f1
                self.no_improvement_num = 0
                # only when best slot_f1 is found, we save the model
                self.save_model()
            else:
                self.no_improvement_num += 1
                logger.info("No better model found (%d/%d)" % (self.no_improvement_num, self.early_stop))
        
        if self.no_improvement_num >= self.early_stop:
            self.stop_training_flag = True
        
        return intent_acc, slot_f1, self.stop_training_flag
    
    def save_model(self):
        saved_path = os.path.join(self.dump_path, "best_model.pth")
        torch.save(self.slu_model, saved_path)

        logger.info("Best model has been saved to %s" % saved_path)

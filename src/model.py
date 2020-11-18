
import torch
import torch.nn as nn
from src.modules import Lstm, Lstm4pretr, IntentPredictor, SlotPredictor, UniformDistriGen, LabelEncoder

class ModelSLU(nn.Module):
    def __init__(self, params, vocab_en, vocab_trans):
        super(ModelSLU, self).__init__()
        self.label_reg = params.la_reg
        self.adversarial = params.adv
        self.intent_adv = params.intent_adv
        self.zeroshot = params.zs
        if self.label_reg == True:
            self.label_encoder = LabelEncoder(params)
        self.lstm = Lstm(params, vocab_en, vocab_trans)
        self.intent_predictor = IntentPredictor(params)
        self.slot_predictor = SlotPredictor(params)
    
    def forward(self, x_en, lengths_en, x_trans, lengths_trans, padded_y2_en=None, padded_y2_trans=None, istestset=False):
        if x_en is not None:
            lstm_hidden_en = self.lstm(x_en, input_type="en")
            if self.adversarial == True and self.training == True:
                slot_pred_en, latent_variable_en = self.slot_predictor(lstm_hidden_en)
            else:
                slot_pred_en = self.slot_predictor(lstm_hidden_en)

            if self.label_reg == False or self.training == False:
                if self.intent_adv == True:
                    intent_pred_en, intent_lv_en = self.intent_predictor(lstm_hidden_en, lengths_en)
                else:
                    intent_pred_en = self.intent_predictor(lstm_hidden_en, lengths_en)
            else:
                if self.intent_adv == True:
                    intent_pred_en, utter_repre_en, intent_lv_en = self.intent_predictor(lstm_hidden_en, lengths_en)
                else:
                    intent_pred_en, utter_repre_en = self.intent_predictor(lstm_hidden_en, lengths_en)
                label_repre_en = self.label_encoder(padded_y2_en, lengths_en)
        else:
            intent_pred_en, slot_pred_en = None, None
        
        if self.zeroshot == True and self.training == True:
            # for zero-shot training
            lstm_hidden_trans = self.lstm(x_trans, input_type="en")
        else:
            # zero-shot validation and test
            lstm_hidden_trans = self.lstm(x_trans, input_type="trans")
        if self.adversarial == True and self.training == True:
            slot_pred_trans, latent_variable_trans = self.slot_predictor(lstm_hidden_trans)
        else:
            slot_pred_trans = self.slot_predictor(lstm_hidden_trans)

        if self.label_reg == False or self.training == False:
            if self.intent_adv == True:
                intent_pred_trans, intent_lv_trans = self.intent_predictor(lstm_hidden_trans, lengths_trans)
            else:
                intent_pred_trans = self.intent_predictor(lstm_hidden_trans, lengths_trans)
            if self.adversarial == True and self.training == True:
                if self.intent_adv == True:
                    return intent_pred_en, slot_pred_en, intent_pred_trans, slot_pred_trans, latent_variable_en, latent_variable_trans, intent_lv_en, intent_lv_trans
                else:
                    return intent_pred_en, slot_pred_en, intent_pred_trans, slot_pred_trans, latent_variable_en, latent_variable_trans
            else:
                return intent_pred_en, slot_pred_en, intent_pred_trans, slot_pred_trans
        else:
            if self.intent_adv == True:
                intent_pred_trans, utter_repre_trans, intent_lv_trans = self.intent_predictor(lstm_hidden_trans, lengths_trans)
            else:
                intent_pred_trans, utter_repre_trans = self.intent_predictor(lstm_hidden_trans, lengths_trans)
            label_repre_trans = self.label_encoder(padded_y2_trans, lengths_trans)

            if self.adversarial == True and self.training == True:
                if self.intent_adv == True:
                    return intent_pred_en, slot_pred_en, intent_pred_trans, slot_pred_trans, utter_repre_en, label_repre_en, utter_repre_trans, label_repre_trans, latent_variable_en, latent_variable_trans, intent_lv_en, intent_lv_trans
                else:
                    return intent_pred_en, slot_pred_en, intent_pred_trans, slot_pred_trans, utter_repre_en, label_repre_en, utter_repre_trans, label_repre_trans, latent_variable_en, latent_variable_trans
            else:
                return intent_pred_en, slot_pred_en, intent_pred_trans, slot_pred_trans, utter_repre_en, label_repre_en, utter_repre_trans, label_repre_trans


class ModelSLU4Adv(nn.Module):
    '''
    ModelSLU for adversarial training
    '''
    def __init__(self, params, vocab_en, vocab_trans):
        super(ModelSLU4Adv, self).__init__()
        self.label_reg = params.la_reg
        self.intent_adv = params.intent_adv
        self.model = ModelSLU(params, vocab_en, vocab_trans)
        self.uniform_distri_gen = UniformDistriGen(params)

    def forward(self, x_en, lengths_en, x_trans, lengths_trans, padded_y2_en=None, padded_y2_trans=None, istestset=False):
        if self.label_reg == False or self.training == False:
            if self.training == True:
                if self.intent_adv == True:
                    intent_pred_en, slot_pred_en, intent_pred_trans, slot_pred_trans, latent_variable_en, latent_variable_trans, intent_lv_en, intent_lv_trans = self.model(x_en, lengths_en, x_trans, lengths_trans, padded_y2_en, padded_y2_trans, istestset)

                    adv_pred_en, adv_pred_intent_en = self.uniform_distri_gen(latent_variable_en, intent_lv_en)
                    adv_pred_trans, adv_pred_intent_trans = self.uniform_distri_gen(latent_variable_trans, intent_lv_trans)

                    return intent_pred_en, slot_pred_en, adv_pred_en, adv_pred_intent_en, intent_pred_trans, slot_pred_trans, adv_pred_trans, adv_pred_intent_trans
                else:
                    intent_pred_en, slot_pred_en, intent_pred_trans, slot_pred_trans, latent_variable_en, latent_variable_trans = self.model(x_en, lengths_en, x_trans, lengths_trans, padded_y2_en, padded_y2_trans, istestset)

                    adv_pred_en = self.uniform_distri_gen(latent_variable_en)
                    adv_pred_trans = self.uniform_distri_gen(latent_variable_trans)
                    
                    return intent_pred_en, slot_pred_en, adv_pred_en, intent_pred_trans, slot_pred_trans, adv_pred_trans
            else:
                intent_pred_en, slot_pred_en, intent_pred_trans, slot_pred_trans = self.model(x_en, lengths_en, x_trans, lengths_trans, padded_y2_en, padded_y2_trans, istestset)

                return intent_pred_en, slot_pred_en, intent_pred_trans, slot_pred_trans
        else:
            # must in the training mode
            if self.intent_adv == True:
                intent_pred_en, slot_pred_en, intent_pred_trans, slot_pred_trans, utter_repre_en, label_repre_en, utter_repre_trans, label_repre_trans, latent_variable_en, latent_variable_trans, intent_lv_en, intent_lv_trans = self.model(x_en, lengths_en, x_trans, lengths_trans, padded_y2_en, padded_y2_trans, istestset)

                adv_pred_en, adv_pred_intent_en = self.uniform_distri_gen(latent_variable_en, intent_lv_en)
                adv_pred_trans, adv_pred_intent_trans = self.uniform_distri_gen(latent_variable_trans, intent_lv_trans)

                return intent_pred_en, slot_pred_en, adv_pred_en, adv_pred_intent_en, intent_pred_trans, slot_pred_trans, adv_pred_trans, adv_pred_intent_trans, utter_repre_en, label_repre_en, utter_repre_trans, label_repre_trans

            else:
                intent_pred_en, slot_pred_en, intent_pred_trans, slot_pred_trans, utter_repre_en, label_repre_en, utter_repre_trans, label_repre_trans, latent_variable_en, latent_variable_trans = self.model(x_en, lengths_en, x_trans, lengths_trans, padded_y2_en, padded_y2_trans, istestset)
            
                adv_pred_en = self.uniform_distri_gen(latent_variable_en)
                adv_pred_trans = self.uniform_distri_gen(latent_variable_trans)

                return intent_pred_en, slot_pred_en, adv_pred_en, intent_pred_trans, slot_pred_trans, adv_pred_trans, utter_repre_en, label_repre_en, utter_repre_trans, label_repre_trans


class ModelSLU4Pretr(nn.Module):
    def __init__(self, params, vocab_en):
        super(ModelSLU4Pretr, self).__init__()
        self.label_reg = params.la_reg
        self.adversarial = params.adv
        self.intent_adv = params.intent_adv
        self.label_encoder = LabelEncoder(params)
        self.lstm = Lstm4pretr(params, vocab_en)
        self.intent_predictor = IntentPredictor(params)
        self.slot_predictor = SlotPredictor(params)
    
    def forward(self, x_1, lengths_1, x_2, lengths_2, padded_y2_1, padded_y2_2):
        # sample 1
        lstm_hidden_1 = self.lstm(x_1)
        slot_pred_1 = self.slot_predictor(lstm_hidden_1)
        if self.adversarial == True:
            slot_pred_1 = slot_pred_1[0]
        if self.intent_adv == True:
            intent_pred_1, utter_repre_1, _ = self.intent_predictor(lstm_hidden_1, lengths_1)
        else:
            intent_pred_1, utter_repre_1 = self.intent_predictor(lstm_hidden_1, lengths_1)
        label_repre_1 = self.label_encoder(padded_y2_1, lengths_1)

        # sample2
        lstm_hidden_2 = self.lstm(x_2)
        slot_pred_2 = self.slot_predictor(lstm_hidden_2)
        if self.adversarial == True:
            slot_pred_2 = slot_pred_2[0]
        if self.intent_adv == True:
            intent_pred_2, utter_repre_2, _ = self.intent_predictor(lstm_hidden_2, lengths_2)
        else:
            intent_pred_2, utter_repre_2 = self.intent_predictor(lstm_hidden_2, lengths_2)
        label_repre_2 = self.label_encoder(padded_y2_2, lengths_2)

        return intent_pred_1, slot_pred_1, intent_pred_2, slot_pred_2, utter_repre_1, label_repre_1, utter_repre_2, label_repre_2
    
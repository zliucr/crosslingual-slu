
from config import get_params
from src.utils import init_experiment
from src.data_loader import get_dataloader, dataloader4pretr
from src.model import ModelSLU, ModelSLU4Adv, ModelSLU4Pretr
from src.trainer import SLUTrainer
import random
import torch

import numpy as np
from tqdm import tqdm
import os

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main(params):
    # initialize experiment
    logger = init_experiment(params, logger_filename=params.logger_filename)
    
    if params.pretr_la_enc == True and params.ckpt_labelenc == "":
        dataloader_pretr, vocab_en = dataloader4pretr(params)
        slu_model4pretr = ModelSLU4Pretr(params, vocab_en)
        slu_model4pretr.cuda()
        trainer4pretr = SLUTrainer(params, slu_model4pretr, pretrain_flag=True)

        # pretraining label encoder
        logger.info("============== Pretraining Label Encoder ==============")
        for e in range(params.pretr_epoch):
            logger.info("============== epoch %d ==============" % e)
            pbar = tqdm(enumerate(dataloader_pretr), total=len(dataloader_pretr))
            intent_loss_list, slot_loss_list, cos_loss_list = [], [], []
            for i, (x_1, padded_y2_1, lengths_1, y1_1, y2_1, x_2, padded_y2_2, lengths_2, y1_2, y2_2) in pbar:
                x_1, lengths_1, y1_1 = x_1.cuda(), lengths_1.cuda(), y1_1.cuda()
                x_2, lengths_2, y1_2 = x_2.cuda(), lengths_2.cuda(), y1_2.cuda()

                padded_y2_1, padded_y2_2 = padded_y2_1.cuda(), padded_y2_2.cuda()
                intent_loss_1, slot_loss_1, intent_loss_2, slot_loss_2, cos_loss = trainer4pretr.pretr_label_encoder_step(e, x_1, lengths_1, x_2, lengths_2, y1_1, y2_1, y1_2, y2_2, padded_y2_1, padded_y2_2)
                
                intent_loss_list.append(intent_loss_1)
                intent_loss_list.append(intent_loss_2)
                slot_loss_list.append(slot_loss_1)
                slot_loss_list.append(slot_loss_2)
                cos_loss_list.append(cos_loss)

                pbar.set_description("(Epoch {}) INTENT:{:.3f} SLOT:{:.3f} COS:{:.3f}".format((e+1), np.mean(intent_loss_list), np.mean(slot_loss_list), np.mean(cos_loss_list)))

            logger.info("(Finished epoch {}) INTENT:{:.3f} SLOT:{:.3f} COS:{:.3f}".format((e+1), np.mean(intent_loss_list), np.mean(slot_loss_list), np.mean(cos_loss_list)))
        
        label_encoder_saved_path = os.path.join(params.dump_path, "label_encoder.pth")
        logger.info("Saving label encoder to %s" % label_encoder_saved_path)
        torch.save(slu_model4pretr.label_encoder, label_encoder_saved_path)


    # get dataloader and vocabulary
    dataloader_tr, dataloader_val, dataloader_test, vocab_en, vocab_trans = get_dataloader(params)

    # build model
    if params.adv == True:
        slu_model = ModelSLU4Adv(params, vocab_en, vocab_trans)
    else:
        slu_model = ModelSLU(params, vocab_en, vocab_trans)
    slu_model.cuda()
    
    if params.pretr_la_enc == True:
        # copy label encoder
        if params.adv == True:
            if params.ckpt_labelenc != "":
                logger.info("Loading label encoder from %s" % params.ckpt_labelenc)
                pretrained_label_encoder = torch.load(params.ckpt_labelenc)
                pretrained_label_encoder = pretrained_label_encoder.cuda()
                slu_model.model.label_encoder = pretrained_label_encoder
            else:
                slu_model.model.label_encoder = slu_model4pretr.label_encoder
        else:
            if params.ckpt_labelenc != "":
                logger.info("Loading label encoder from %s" % params.ckpt_labelenc)
                pretrained_label_encoder = torch.load(params.ckpt_labelenc)
                pretrained_label_encoder = pretrained_label_encoder.cuda()
                slu_model.label_encoder = pretrained_label_encoder
            else:
                slu_model.label_encoder = slu_model4pretr.label_encoder
        
    def get_learnable_params(module):
        return [p for p in module.parameters() if p.requires_grad]

    model_params = get_learnable_params(slu_model)
    print("model parameters: %d" % sum(p.numel() for p in model_params))
    # build trainer
    slu_trainer = SLUTrainer(params, slu_model)

    logger.info("============== Start training ==============")
    for e in range(params.epoch):
        logger.info("============== epoch %d ==============" % e)
        
        pbar = tqdm(enumerate(dataloader_tr), total=len(dataloader_tr))
        if params.tar_only == False:
            
            intent_loss_en_list, slot_loss_en_list, intent_loss_trans_list, slot_loss_trans_list, cos_loss_list = [], [], [], [], []
            if params.adv == True:
                model_en_adv_loss_list, model_trans_adv_loss_list, udg_en_adv_loss_list, udg_trans_adv_loss_list = [], [], [], []
                if params.intent_adv == True:
                    model_en_adv_intent_loss_list, model_trans_adv_intent_loss_list, udg_en_adv_intent_loss_list, udg_trans_adv_intent_loss_list = [], [], [], []

            for i, (x_en, padded_y2_en, lengths_en, y1_en, y2_en, x_trans, padded_y2_trans, lengths_trans, y1_trans, y2_trans) in pbar:
                x_en, lengths_en, y1_en = x_en.cuda(), lengths_en.cuda(), y1_en.cuda()
                x_trans, lengths_trans, y1_trans = x_trans.cuda(), lengths_trans.cuda(), y1_trans.cuda()
                if params.la_reg == False:
                    if params.adv == True:
                        # adversarial lvm
                        if params.intent_adv == True:
                            intent_loss_en, slot_loss_en, intent_loss_trans, slot_loss_trans, model_en_adv_loss, model_trans_adv_loss, udg_en_adv_loss, udg_trans_adv_loss, model_en_adv_intent_loss, model_trans_adv_intent_loss, udg_en_adv_intent_loss, udg_trans_adv_intent_loss = slu_trainer.joint_train_step(e, x_en, lengths_en, x_trans, lengths_trans, y1_en, y2_en, y1_trans, y2_trans)

                            model_en_adv_intent_loss_list.append(model_en_adv_intent_loss)
                            model_trans_adv_intent_loss_list.append(model_trans_adv_intent_loss)
                            udg_en_adv_intent_loss_list.append(udg_en_adv_intent_loss)
                            udg_trans_adv_intent_loss_list.append(udg_trans_adv_intent_loss)
                        else:
                            intent_loss_en, slot_loss_en, intent_loss_trans, slot_loss_trans, model_en_adv_loss, model_trans_adv_loss, udg_en_adv_loss, udg_trans_adv_loss = slu_trainer.joint_train_step(e, x_en, lengths_en, x_trans, lengths_trans, y1_en, y2_en, y1_trans, y2_trans)

                        model_en_adv_loss_list.append(model_en_adv_loss)
                        model_trans_adv_loss_list.append(model_trans_adv_loss)
                        udg_en_adv_loss_list.append(udg_en_adv_loss)
                        udg_trans_adv_loss_list.append(udg_trans_adv_loss)
                    else:
                        intent_loss_en, slot_loss_en, intent_loss_trans, slot_loss_trans = slu_trainer.joint_train_step(e, x_en, lengths_en, x_trans, lengths_trans, y1_en, y2_en, y1_trans, y2_trans)
                else:
                    padded_y2_en, padded_y2_trans = padded_y2_en.cuda(), padded_y2_trans.cuda()
                    if params.adv == True:
                        # adversarial lvm
                        if params.intent_adv == True:
                            intent_loss_en, slot_loss_en, intent_loss_trans, slot_loss_trans, cos_loss, model_en_adv_loss, model_trans_adv_loss, udg_en_adv_loss, udg_trans_adv_loss, model_en_adv_intent_loss, model_trans_adv_intent_loss, udg_en_adv_intent_loss, udg_trans_adv_intent_loss = slu_trainer.joint_train_step(e, x_en, lengths_en, x_trans, lengths_trans, y1_en, y2_en, y1_trans, y2_trans, padded_y2_en, padded_y2_trans)

                            model_en_adv_intent_loss_list.append(model_en_adv_intent_loss)
                            model_trans_adv_intent_loss_list.append(model_trans_adv_intent_loss)
                            udg_en_adv_intent_loss_list.append(udg_en_adv_intent_loss)
                            udg_trans_adv_intent_loss_list.append(udg_trans_adv_intent_loss)
                        else:
                            intent_loss_en, slot_loss_en, intent_loss_trans, slot_loss_trans, cos_loss, model_en_adv_loss, model_trans_adv_loss, udg_en_adv_loss, udg_trans_adv_loss = slu_trainer.joint_train_step(e, x_en, lengths_en, x_trans, lengths_trans, y1_en, y2_en, y1_trans, y2_trans, padded_y2_en, padded_y2_trans)

                        model_en_adv_loss_list.append(model_en_adv_loss)
                        model_trans_adv_loss_list.append(model_trans_adv_loss)
                        udg_en_adv_loss_list.append(udg_en_adv_loss)
                        udg_trans_adv_loss_list.append(udg_trans_adv_loss)
                    else:
                        intent_loss_en, slot_loss_en, intent_loss_trans, slot_loss_trans, cos_loss = slu_trainer.joint_train_step(e, x_en, lengths_en, x_trans, lengths_trans, y1_en, y2_en, y1_trans, y2_trans, padded_y2_en, padded_y2_trans)
                    
                    cos_loss_list.append(cos_loss)

                intent_loss_en_list.append(intent_loss_en)
                slot_loss_en_list.append(slot_loss_en)
                intent_loss_trans_list.append(intent_loss_trans)
                slot_loss_trans_list.append(slot_loss_trans)

                if params.la_reg == False:
                    if params.adv == True:
                        if params.intent_adv == True:
                            pbar.set_description("(E{})I1:{:.1f}S1:{:.1f}I2:{:.1f}S2:{:.1F}G1:{:.2f}G2:{:.2f}G3:{:.2f}G4:{:.2f}D1:{:.2f}D2:{:.2f}D3:{:.2f}D4:{:.2f}".format((e+1), np.mean(intent_loss_en_list), np.mean(slot_loss_en_list), np.mean(intent_loss_trans_list), np.mean(slot_loss_trans_list), np.mean(model_en_adv_loss_list), np.mean(model_trans_adv_loss_list), np.mean(model_en_adv_intent_loss_list), np.mean(model_trans_adv_intent_loss_list), np.mean(udg_en_adv_loss_list), np.mean(udg_trans_adv_loss_list), np.mean(udg_en_adv_intent_loss_list), np.mean(udg_trans_adv_intent_loss_list)))
                        else:
                            pbar.set_description("(E{}) I1:{:.4f} S1:{:.4f} I2:{:.4f} S2:{:.4F} G1:{:.4f} G2:{:.4f} D1:{:.4f} D2:{:.4f}".format((e+1), np.mean(intent_loss_en_list), np.mean(slot_loss_en_list), np.mean(intent_loss_trans_list), np.mean(slot_loss_trans_list), np.mean(model_en_adv_loss_list), np.mean(model_trans_adv_loss_list), np.mean(udg_en_adv_loss_list), np.mean(udg_trans_adv_loss_list)))
                    else:
                        pbar.set_description("(Epoch {}) EN_INTENT:{:.4f} EN_SLOT:{:.4f} TRANS_INTENT:{:.4f} TRANS_SLOT:{:.4F}".format((e+1), np.mean(intent_loss_en_list), np.mean(slot_loss_en_list), np.mean(intent_loss_trans_list), np.mean(slot_loss_trans_list)))
                else:
                    if params.adv == True:
                        if params.intent_adv == True:
                            pbar.set_description("(E{})I1:{:.1f}S1:{:.1f}I2:{:.1f}S2:{:.1F}C:{:.1f}G1:{:.2f}G2:{:.2f}G3:{:.2f}G4:{:.2f}D1:{:.2f}D2:{:.2f}D3:{:.2f}D4:{:.2f}".format((e+1), np.mean(intent_loss_en_list), np.mean(slot_loss_en_list), np.mean(intent_loss_trans_list), np.mean(slot_loss_trans_list), np.mean(cos_loss_list), np.mean(model_en_adv_loss_list), np.mean(model_trans_adv_loss_list), np.mean(model_en_adv_intent_loss_list), np.mean(model_trans_adv_intent_loss_list), np.mean(udg_en_adv_loss_list), np.mean(udg_trans_adv_loss_list), np.mean(udg_en_adv_intent_loss_list), np.mean(udg_trans_adv_intent_loss_list)))
                        else:
                            pbar.set_description("(E{}) I1:{:.3f} S1:{:.3f} I2:{:.3f} S2:{:.3f} C:{:.3f} G1:{:.3f} G2:{:.3f} D1:{:.3f} D2:{:.3f}".format((e+1), np.mean(intent_loss_en_list), np.mean(slot_loss_en_list), np.mean(intent_loss_trans_list), np.mean(slot_loss_trans_list), np.mean(cos_loss_list), np.mean(model_en_adv_loss_list), np.mean(model_trans_adv_loss_list), np.mean(udg_en_adv_loss_list), np.mean(udg_trans_adv_loss_list)))
                    else:
                        pbar.set_description("(Epoch {}) EN_INTENT:{:.3f} EN_SLOT:{:.3f} TR_INTENT:{:.3f} TR_SLOT:{:.3f} COS:{:.3f}".format((e+1), np.mean(intent_loss_en_list), np.mean(slot_loss_en_list), np.mean(intent_loss_trans_list), np.mean(slot_loss_trans_list), np.mean(cos_loss_list)))
        else:
            intent_loss_list, slot_loss_list = [], []
            for i, (X, lengths, y1, y2) in pbar:
                X, lengths, y1 = X.cuda(), lengths.cuda(), y1.cuda()
                intent_loss, slot_loss = slu_trainer.single_train_step(X, lengths, y1, y2)

                intent_loss_list.append(intent_loss)
                slot_loss_list.append(slot_loss)

                pbar.set_description("(Epoch {}) INTENT LOSS:{:.4f} SLOT LOSS:{:.4f}".format((e+1), np.mean(intent_loss_list), np.mean(slot_loss_list)))
        
        if params.tar_only == False:
            if params.la_reg == False:
                if params.adv == True:
                    if params.intent_adv == True:
                        logger.info("(E{})I1:{:.1f}S1:{:.1f}I2:{:.1f}S2:{:.1f}G1:{:.2f}G2:{:.2f}G3:{:.2f}G4:{:.2f}D1:{:.2f}D2:{:.2f}D3:{:.2f}D4:{:.2f}".format((e+1), np.mean(intent_loss_en_list), np.mean(slot_loss_en_list), np.mean(intent_loss_trans_list), np.mean(slot_loss_trans_list), np.mean(model_en_adv_loss_list), np.mean(model_trans_adv_loss_list), np.mean(model_en_adv_intent_loss_list), np.mean(model_trans_adv_intent_loss_list), np.mean(udg_en_adv_loss_list), np.mean(udg_trans_adv_loss_list), np.mean(udg_en_adv_intent_loss_list), np.mean(udg_trans_adv_intent_loss_list)))
                    else:
                        logger.info("(E{}) I1:{:.4f} S1:{:.4f} I2:{:.4f} S2:{:.4F} G1:{:.4f} G2:{:.4f} D1:{:.4f} D2:{:.4f}".format((e+1), np.mean(intent_loss_en_list), np.mean(slot_loss_en_list), np.mean(intent_loss_trans_list), np.mean(slot_loss_trans_list), np.mean(model_en_adv_loss_list), np.mean(model_trans_adv_loss_list), np.mean(udg_en_adv_loss_list), np.mean(udg_trans_adv_loss_list)))
                else:
                    logger.info("Finish training epoch {} EN_INTENT:{:.4f} EN_SLOT:{:.4f} TRANS_INTENT:{:.4f} TRANS_SLOT:{:.4f}".format((e+1), np.mean(intent_loss_en_list), np.mean(slot_loss_en_list), np.mean(intent_loss_trans_list), np.mean(slot_loss_trans_list)))
            else:
                if params.adv == True:
                    if params.intent_adv == True:
                        logger.info("(E{})I1:{:.1f}S1:{:.1f}I2:{:.1f}S2:{:.1f}C:{:.1f}G1:{:.2f}G2:{:.2f}G3:{:.2f}G4:{:.2f}D1:{:.2f}D2:{:.2f}D3:{:.2f}D4:{:.2f}".format((e+1), np.mean(intent_loss_en_list), np.mean(slot_loss_en_list), np.mean(intent_loss_trans_list), np.mean(slot_loss_trans_list), np.mean(cos_loss_list), np.mean(model_en_adv_loss_list), np.mean(model_trans_adv_loss_list), np.mean(model_en_adv_intent_loss_list), np.mean(model_trans_adv_intent_loss_list), np.mean(udg_en_adv_loss_list), np.mean(udg_trans_adv_loss_list), np.mean(udg_en_adv_intent_loss_list), np.mean(udg_trans_adv_intent_loss_list)))
                    else:
                        logger.info("(E{}) I1:{:.4f} S1:{:.4f} I2:{:.4f} S2:{:.4f} C:{:.4f} G1:{:.4f} G2:{:.4f} D1:{:.4f} D2:{:.4f}".format((e+1), np.mean(intent_loss_en_list), np.mean(slot_loss_en_list), np.mean(intent_loss_trans_list), np.mean(slot_loss_trans_list), np.mean(cos_loss_list), np.mean(model_en_adv_loss_list), np.mean(model_trans_adv_loss_list), np.mean(udg_en_adv_loss_list), np.mean(udg_trans_adv_loss_list)))
                else:
                    logger.info("Finish training epoch {} EN_INTENT:{:.3f} EN_SLOT:{:.3f} TR_INTENT:{:.3f} TR_SLOT:{:.3f} COS:{:.3f}".format((e+1), np.mean(intent_loss_en_list), np.mean(slot_loss_en_list), np.mean(intent_loss_trans_list), np.mean(slot_loss_trans_list), np.mean(cos_loss_list)))
        else:
            logger.info("Finish training epoch {} INTENT LOSS:{:.4f} SLOT LOSS:{:.4f}".format((e+1),np.mean(intent_loss_list), np.mean(slot_loss_list)))

        logger.info("============== Evaluate Epoch {} ==============".format(e+1))
        intent_acc, slot_f1, stop_training_flag = slu_trainer.evaluate(dataloader_val, istestset=False)

        logger.info("Dev Set: Intent ACC:{:.4f} (Best Acc:{:.4f}). Slot F1:{:.4f}. (Best F1:{:.4f})".format(intent_acc, slu_trainer.best_intent_acc, slot_f1, slu_trainer.best_slot_f1))

        intent_acc, slot_f1, _ = slu_trainer.evaluate(dataloader_test, istestset=True)

        logger.info("Test set: Intent ACC:{:.4f} (Best Acc:{:.4f}). Slot F1:{:.4f}. (Best F1:{:.4f})".format(intent_acc, slu_trainer.best_intent_acc, slot_f1, slu_trainer.best_slot_f1))

        if stop_training_flag == True:
            break

if __name__ == "__main__":
    params = get_params()
    
    random_seed(params.seed)
    main(params)

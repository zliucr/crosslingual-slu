import argparse

def get_params():
    # parse parameters
    parser = argparse.ArgumentParser(description="Multilingual Tasks")
    parser.add_argument("--exp_name", type=str, default="default", help="Experiment name")
    parser.add_argument("--logger_filename", type=str, default="multilingual_tasks.log")
    parser.add_argument("--dump_path", type=str, default="experiments", help="Experiment saved root path")
    parser.add_argument("--exp_id", type=str, default="1", help="Experiment id")

    parser.add_argument("--trans_lang", type=str, default="es", help="Choose a language to transfer (es, th)")
    parser.add_argument("--emb_file_en", type=str, default="./emb/refine.en.align.en-es.vec", help="Path of word embeddings for English")
    parser.add_argument("--emb_file_trans", type=str, default="./emb/refine.es.align.en-es.vec", help="Path of word embeddings for transfer language")

    # model parameters
    parser.add_argument("--bidirection", default=False, action="store_true", help="Bidirectional lstm")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for lstm")
    parser.add_argument("--n_layer", type=int, default=2, help="Number of lstm layer")
    parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=250, help="Hidden layer dimension")

    # train parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epoch", type=int, default=300, help="Number of epoch")
    parser.add_argument("--num_iter", type=int, default=3000, help="Number of iterations")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--clean_txt", default=False, action="store_true", help="clean data")
    parser.add_argument("--early_stop", type=int, default=5, help="No improvement after several epoch, we stop training")

    # target only
    parser.add_argument("--tar_only", default=False, action="store_true", help="consider target language only")

    # trim data
    parser.add_argument("--trim", default=False, action="store_true", help="trim data")
    parser.add_argument("--prop", type=float, default=0.01, help="trimed data size")

    # data statistic
    parser.add_argument("--num_intent", type=int, default=12, help="Number of intent in the dataset")
    parser.add_argument("--num_slot", type=int, default=22, help="Number of slot in the dataset")
    
    # label regularization
    parser.add_argument("--la_reg", default=False, action="store_true", help="label regularzation")
    parser.add_argument("--pretr_la_enc", default=False, action="store_true", help="pretrain label encoder")
    parser.add_argument("--pretr_epoch", type=int, default=2, help="number of epoch for pretraining")
    parser.add_argument("--n_layer_la_enc", type=int, default=1, help="number of layers for label encoder")
    parser.add_argument("--emb_dim_la_enc", type=int, default=100, help="embedding dimension for label encoder")
    parser.add_argument("--hidden_dim_la_enc", type=int, default=150, help="hidden dimension for label encoder")

    # lvm
    parser.add_argument("--lvm", default=False, action="store_true", help="latent variable model")
    parser.add_argument("--lvm_dim", type=int, default=200, help="lvm dimension")
    parser.add_argument("--rmcrf", default=False, action="store_true", help="remove crf")
    parser.add_argument("--scale", default=False, action="store_true", help="smaller Enlgish train set in on epoch")
    parser.add_argument("--scale_size", type=float, default=0.25, help="scale size")
    parser.add_argument("--adv", default=False, action="store_true", help="use adversarial training for lvm for slot")
    parser.add_argument("--epoch_patient", type=int, default=1, help="It decides when to start training the lvm in the adversarial training")
    parser.add_argument("--intent_adv", default=False, action="store_true", help="adversarial training for lvm in intent")

    # Gussian noise
    parser.add_argument("--embnoise", default=False, action="store_true", help="add gaussian noise")

    # zero-shot
    parser.add_argument("--zs", default=False, action="store_true", help="zero-shot adaptation")

    # random seed
    parser.add_argument("--seed", type=int, default=555, help="random seed (three seeds: 555, 666, 777)")
    
    # label encoder checkpoint
    parser.add_argument("--ckpt_labelenc", type=str, default="", help="checkpoint for the label encoder")

    # reload model
    parser.add_argument("--model_path", type=str, default="", help="model path for reloading")

    params = parser.parse_args()

    return params

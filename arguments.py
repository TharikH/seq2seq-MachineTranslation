import argparse

def parsArg():
    parser = argparse.ArgumentParser()

    parser.add_argument('-wp','--wandb_project',default='dl-assignment-3-final',
                        help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we','--wandb_entity',default='cs22m058',
                        help='wandb entity name')
    parser.add_argument('-e','--epochs',default=20,type=int,
                        help='Number of epochs to train neural network')
    parser.add_argument('-b','--batch_size',default=256, type=int,
                        help='Batch size used to train neural network')
    parser.add_argument('-l','--loss',default='cross_entropy',choices = ["cross_entropy"],
                        help='choices: ["cross_entropy"]')
    parser.add_argument('-nel','--num_encoder_layers',default=3, type=int,
                        help='Number of hidden layers used in encoder')
    parser.add_argument('-ndl','--num_decoder_layers',default=2, type=int,
                        help='Number of hidden layers used in decoder')
    parser.add_argument('-eh','--encoder_hidden_size',default=256, type=int,
                        help='Number of hidden neurons in encoder')
    parser.add_argument('-dh','--decoder_hidden_size',default=256, type=int,
                        help='Number of hidden neurons in decoder')
    parser.add_argument('-bd','--bidirectional',default=True, type=bool,
                        help='Need bidirection or not')
    parser.add_argument('-ee','--encoder_embedding',default=256, type=int,
                        help='Dimension for input embedding')
    parser.add_argument('-de','--decoder_embedding',default=256, type=int,
                        help='Dimension for output embedding')
    parser.add_argument('-d','--dropout',default=0.2, type=float,
                        help='percentage of dropout needed')
    parser.add_argument('-c','--cell_type',default='lstm', choices = ["lstm","rnn","gru"],
                        help='Which cell type to execute')
    parser.add_argument('-bw','--beam_width',default=1, type=int,
                        help='beam search beam width')
    parser.add_argument('-a','--use_attention',default=False, type=bool,
                        help='Need Attention or not')
    
    args = parser.parse_args()
    
    return args
from datetime import datetime
import torch
import argparse
import os
import pdb

#model=make_model(input_size=1024, num_layers=3, hidden_size=1024, num_cells=1024, dropout_rate=0.0,n_blocks=4, n_hidden=3, prediction_step=99, mu_size=6)
class Args:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        #training process
        ## wathc out the behavior here, if specify token, it will use token. if specify sample_flag, it will not use sample_flag
        self.parser.add_argument('--sample_flag', action = 'store_false', help = 'sample from the distribution or mean')
        self.parser.add_argument('--token', action = 'store_true', help = 'parameter token')
        self.parser.add_argument('--modelref', default=10086, help='reference for model, no physical meaning')
        self.parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu',
                                 help='cuda:[d] | cpu')
        #self.parser.add_argument('--dataset', default="cylinder",
                                 #help='Select datase -- cyliner| sonic| vas')
        self.parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
        self.parser.add_argument('--epoch', type = int, default = 400000, help = 'training epoch')
        self.parser.add_argument('--batchsize', type = int, default = '64', help = 'batchsize')
        #model
        self.parser.add_argument('--dataset', type = str, default = 'cylinder', help = 'training dataset, sine, cylinder, KS')
        self.parser.add_argument('--noise', type = float, default = 0.0, help = 'noise level')
        
        # train or test
        self.parser.add_argument('--train', type = int, default = 1, help = '1: train, 0: test')
        ##
        ## test_options
        self.parser.add_argument('--test_epoch', type = int, default = 450000, help = 'test epoch')
        self.parser.add_argument('--test_samples', type = int, default = 1, help = 'test samples')

        self.parser.add_argument('--history_length', type = int, default = 300, help = 'given history length used in test mode')
        self.parser.add_argument('--prediction_length', type = int, default = 100, help = 'given future length used in test mode')
        
        ## hyperparameters
        #input_size, num_layers, hidden_size, num_cells, dropout_rate,n_blocks, n_hidden, prediction_step, mu_size,device="cuda:0"):
        ##
        ## input_size=1024, num_layers=3, hidden_size=1024, num_cells=1024, dropout_rate=0.0,n_blocks=4, n_hidden=3, prediction_step=99, mu_size=6
        self.parser.add_argument('--input_size', type=int, default=1024, help='input size for rnn')
        self.parser.add_argument('--num_layers', type = int, default = 2, help = 'num layers for rnn')
        self.parser.add_argument('--hidden_size', type = int, default = 1024, help = 'hidden size for flow')
        self.parser.add_argument('--num_cells', type = int, default = 1024, help = 'hidden size for rnn')
        self.parser.add_argument('--dropout_rate', type = float, default = 0.0, help = 'dropout_rate for physics flow')
        self.parser.add_argument('--n_blocks', type = int, default = 4, help = 'number of blocks for flow')
        self.parser.add_argument('--n_hidden', type = int, default = 2, help = 'number of hidden layers for flow')
        self.parser.add_argument('--conditioning_length', type = int, default = 1000, help = 'conditionining length for flow')
        #self.parser.add_argument('--prediction_step', type = int, default = 99, help = 'prediction step for physics flow')
        self.parser.add_argument('--mu_size', type = int, default = 3, help = 'number of parameters for parametric cases')
        self.parser.add_argument('--adj_var', type = float, default = 1, help = 'varaince for the distribution in latent space')
        #pdb.set_trace()
        ## test p[topms]

        ## transformer hyperparmeters

        self.parser.add_argument('--d_model', type = int, default = 1024, help = 'd_model for transformer')
        self.parser.add_argument('--num_heads', type = int, default = 8, help = 'num heads for transformer')
        self.parser.add_argument('--dim_feedforward_scale', type = int, default = 4, help = 'feed_forward scale')
        self.parser.add_argument('--num_encoder_layers', type = int, default = 3, help = 'num encoder layers')
        self.parser.add_argument('--num_decoder_layers', type = int, default = 3, help = 'num decoder layers')
        #d_model = 32, num_heads = 8, dim_feedforward_scale = 4, act_type = "gelu", num_encoder_layers = 3, num_decoder_layers = 3

        ## 
        self.parser.add_argument('--transfer_flag', type = int, default = 0, help = 'if use fine tuning')
        self.parser.add_argument('--transfer_epoch', type = int, default = 240000, help = 'starting point for fine-tuning')
        self.parser.add_argument('--startT', type = int, default = 0, help = 'start point for data')
        self.parser.add_argument('--shuffle', type = str, default = True, help = 'if shuffle the dataset')
        self.args = self.parser.parse_args()

        

    def update_args(self):
        return self.args 
        





import torch
from sequential_flow import PhysicsFlow, PhysicsFlow_residual, PhysicsFlow_plus_linear, lsunPhysicsFlow, lsunParaPhysicsFlow, lsunParaSinePhysicsFlow, lsunTransformerParaPhysicsFlow, xuRNNParaPhysicsFlow
from sequential_flow import CylinderTransformerParaPhysicsFlow
#from sequential_flow import hgaoTransformerParaPhysicsFlow
from sequential_flow import xu_transformer
import pdb
import numpy as np
from dataclasses import dataclass

            
            

def make_model(input_size, num_layers, hidden_size, num_cells, dropout_rate,n_blocks, n_hidden, mu_size, adj_var = 1, device="cuda:0", conditioning_length = 200,
               d_model = None, num_heads = None, dim_feedforward_scale = None, num_encoder_layers = None, num_decoder_layers = None, config = None, mask_length = None):
    #model = xuRNNParaPhysicsFlow(input_size, num_layers, hidden_size, num_cells, dropout_rate,n_blocks, n_hidden, mu_size, adj_var = adj_var, device = device, conditioning_length = conditioning_length)
    model = xu_transformer(input_size, num_layers, hidden_size, num_cells, dropout_rate, n_blocks, n_hidden, mu_size, adj_var=1, device=device, conditioning_length=conditioning_length)
    # model = CylinderTransformerParaPhysicsFlow(input_size, hidden_size, num_cells, dropout_rate, n_blocks, n_hidden, mu_size, adj_var = 1, device = 'cuda:0', conditioning_length = conditioning_length, 
    #                                     num_heads = num_heads, dim_feedforward_scale = dim_feedforward_scale,
    #                                        num_encoder_layers = num_encoder_layers, num_decoder_layers = num_decoder_layers, mask_length = mask_length)
    #model = lsunTransformerParaPhysicsFlow(input_size, hidden_size, num_cells, dropout_rate, n_blocks, n_hidden, mu_size, adj_var = 1, device = 'cuda:0', conditioning_length = conditioning_length, 
                                        #num_heads = num_heads, dim_feedforward_scale = dim_feedforward_scale,
                                           #num_encoder_layers = num_encoder_layers, num_decoder_layers = num_decoder_layers, mask_length = mask_length, d_model = d_model)#, context_length = 100, prediction_length = 100, d_model = 32, num_heads = 8, dim_feedforward_scale = 4, act_type = "gelu", num_encoder_layers = 3, num_decoder_layers = 3)
    # self, input_size, hidden_size, num_cells, dropout_rate, n_blocks, n_hidden, mu_size, adj_var = 1, device = 'cuda:0', conditioning_length = 200, context_length = 1, prediction_length = 400, config = None)
    #model = hgaoTransformerParaPhysicsFlow(input_size, hidden_size, num_cells, dropout_rate,n_blocks, n_hidden, mu_size, adj_var = adj_var, device = device, conditioning_length = conditioning_length, config = config)
    #model = lsunParaPhysicsFlow(input_size, num_layers, hidden_size, num_cells, dropout_rate,n_blocks, n_hidden, mu_size, adj_var = adj_var, device = device, conditioning_length = conditioning_length)
    #model = lsunParaSinePhysicsFlow(input_size, num_layers, hidden_size, num_cells, dropout_rate,n_blocks, n_hidden, mu_size, adj_var = adj_var, device = device)
    #model = lsunPhysicsFlow(input_size, num_layers, hidden_size, num_cells, dropout_rate,n_blocks, n_hidden, mu_size, device = device)
    return model.to(device)



def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model,path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def generateParameters(dataset, mu_size, device, token = False,train_flag = 1):
    if dataset == "cylinder":
        ## manually switch off the parameter
        #ReAll = torch.from_numpy(np.linspace(300, 1000, 101)).float().to(device).reshape([-1,1])
        #para_mu = ReAll[:mu_size]
        ## previous casees
        if token ==False:
            para_mu = None
        elif token == True:
            ReAll = torch.from_numpy(np.linspace(300, 1000, 101)).float().to(device).reshape([-1,1])
            if train_flag == 1:
                I = [i for i in range(101) if i % 2 == 0]
            elif train_flag == 0:
                #pdb.set_trace()
                I = [i for i in range(101) if i % 2 == 1]
            #pdb.set_trace()
            I = I[:mu_size]
            
            ReAll = ReAll[I]
            para_mu = ReAll
            print('max para is', max(para_mu))
            print('min para is', min(para_mu))
    elif dataset == 'vas2d':        
        if token == False:
            para_mu = None
        elif token == True:
            ReAll = torch.from_numpy(np.linspace(0.3, 0.5, 21)).float().to(device).reshape([-1,1])
            #if train_flag == 1:
                #I = [i for i in range(21) if i % 2 == 0]
            #elif train_flag == 0:
            I = [i for i in range(21) if i % 2 == 1]
            I = I[:mu_size]
            
            ReAll = ReAll[I]
            para_mu = ReAll
            print('max para is', max(para_mu))
            print('min para is', min(para_mu))
    elif dataset == 'mvw':

        if token == False:
            print('mvw no token')
            para_mu = None
        elif token == True:
            print('mvw with token')
            ReAll = torch.from_numpy(np.linspace(273, 373, 101)).float().cuda().reshape([-1,1])
            if train_flag == 1:
                I = [i for i in range(101) if i % 2 == 0]
            elif train_flag == 0:
                I = [i for i in range(101) if i % 2 == 1]
            I = I[:mu_size]
            
            ReAll = ReAll[I]
            para_mu = ReAll
            print('max para is', max(para_mu))
            print('min para is', min(para_mu))
            
    elif dataset == 'les2d':
        para_mu = None
    elif dataset == 'les2d30':
        para_mu = None
    elif dataset == 'sine':
        para_mu = None
    elif dataset == 'stBurgers':
        para_mu = None
    elif dataset == 'stinitial_stBurgers':
        para_mu = None
    elif dataset == 'testsmallstBurgers':
        para_mu = None
    else:
        raise TypeError('Only support cylinder and sine')
    

    return para_mu

# def sample(args, x, model, mu, trajectories, predicted_time_steps):
#     '''
#     x: torch.Tensor(batch_size, initial_time_steps, input_size)
#     mu: torch.Tensoir(batch_size, 1)
#     trajectories: int

#     return samples torch.Tensor(batch_size*trajectories, predicted_time_steps, input_size)
#     '''
#     pass

## test han gao transformer
@dataclass
class SeqSetting:
	n_layer:int = 1
	output_hidden_states:bool = True
	embd_pdrop:float = 0.0
	n_ctx:int = 128 # context length
	n_embd:int = 256 * 4
	layer_norm_epsilon:float = 1e-5
	n_head:int = 4 #16
	attn_pdrop:float = 0.0	
	resid_pdrop:float = 0.0
	activation_function:str = "relu"
	initializer_range:float = 0.02 
	output_attentions:bool = True
	n_epoch:int = 3000000000
	stop_criterion:float = 0.1
	nVarHidden:int = 4
	nnode: int = 256
	paraEnrichDim: int = 6
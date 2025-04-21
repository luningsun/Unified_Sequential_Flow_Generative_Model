import torch
from flow import RealNVP
import torch.nn as nn
from module import MLPDense
import pdb
#from module import enrich_para, Block, Attention, Conv1D, MLP
from module import Block, Attention, Conv1D, MLP, PositionalEncoding
import sys
import time
#sys.path.append('/home/luningsun/storage/gnn_flow/pytorch-ts-master/examples/testutils/')
sys.path.append('./testutils/')
from localpts.modules import FlowOutput
from attention_decoder import AttentionDecoder
#from localpts.modules import StudentOutput, IndependentDistributionOutput
import numpy as np

#This one works for LSTM.
class xuRNNParaPhysicsFlow(nn.Module):
	def __init__(self, input_size, num_layers, hidden_size, num_cells, dropout_rate,n_blocks, n_hidden, mu_size, adj_var = 1, device = 'cuda:0', conditioning_length = 200):
		super(xuRNNParaPhysicsFlow, self).__init__()
		print('use xu token RNN')
		self.rnn = nn.LSTM(
			input_size=input_size,
			hidden_size=num_cells,
			num_layers=num_layers,
			dropout=dropout_rate,
			batch_first=True,
		)
		#pdb.set_trace()
		self.flow = RealNVP(
			input_size=input_size,
			n_blocks=n_blocks,
			hidden_size=hidden_size,
			n_hidden = n_hidden,
			cond_label_size=conditioning_length,
			adj_var = adj_var
		)
		#pdb.set_trace()
		self.num_layers = num_layers
		self.rnn_hidden_size = num_cells
		self.flow_hidden_size = hidden_size
		self.device = device

		target_dim = input_size
		if mu_size is not None:
			#MLPDense(config.paraEnrichDim, config.n_embd,[200, 200], True)
			self.TokenModel = MLPDense(6, num_cells,[200, 200], True)
		
		self.distr_output = FlowOutput(
			self.flow, input_size=target_dim, cond_size=conditioning_length
		)
		self.proj_dist_args = self.distr_output.get_args_proj(num_cells)
		#self.proj_dist_args = self.distr_output.get_args_proj(input_size)

	
	def forward(self, x, mu = None):
		return self.log_likelihood(x,mu)

	def sample(self,x, prediction_length, paraOI = None, sample_flag = True):
		'''
		:param x:torch.tensor(batch_size, 1, input_size)
		:param prediction_length:
		:return:
		'''
		print('sample model')
		#pdb.set_trace()
		outputs = []
		L = self.num_layers
		n_samples = x.size(0)

		label_sec = x.size(1)-1

		if paraOI is not None:
			print('use token')
			
			#paraOI = enrich_para(mu)
			iToken = self.TokenModel(paraOI).unsqueeze(1) #(batch_size, 1, 1024)
			make_up = torch.zeros(L-1, n_samples, self.rnn_hidden_size, dtype=torch.float32).to(self.device)
			h_t = torch.cat([iToken.squeeze(1).unsqueeze(0), make_up], dim=0)
			#pdb.set_trace()
			c_t = torch.zeros(L, n_samples, self.rnn_hidden_size, dtype=torch.float32).to(self.device)
		else:
			h_t = torch.zeros(L, n_samples, self.rnn_hidden_size, dtype=torch.float32).to(self.device)
			c_t = torch.zeros(L, n_samples, self.rnn_hidden_size, dtype=torch.float32).to(self.device)

	
			#     #pdb.set_trace()
			#     #new_samples = self.flow.sample(cond=output)
			#     #outputs.append(new_samples)
		rnn_outputs, (h_t, c_t) = self.rnn(x, (h_t, c_t))
		output = rnn_outputs[:,-1].unsqueeze(1)

		## modify 0506
		distr_args = self.distr_args(rnn_outputs = output)
		new_samples = self.flow.sample(cond = distr_args, sample_flag = sample_flag)


		##s

			#pdb.set_trace()
		future_samples = []
		future_samples.append(new_samples)

		for i in range(prediction_length-1):
			#if i == 0:
				#output, (h_t, c_t) = self.rnn(x[:,-1].unsqueeze(1), (h_t, c_t))
			#else:
			output, (h_t, c_t) = self.rnn(new_samples, (h_t, c_t))

			distr_args = self.distr_args(rnn_outputs = output)
			new_samples = self.flow.sample(cond = distr_args, sample_flag = sample_flag)
			future_samples.append(new_samples)

		future_samples = torch.cat(future_samples, dim=1)

		return future_samples


	def log_likelihood(self,x, paraOI):
		'''
		:param x: torch.tensor(batch_size,prediction_step,input_size)
		:param mu: torch.tensor(batch_size,1)
		:return:
		'''
		n_samples = x.size(0)
		L = self.num_layers

		## 
		#(n,B)
		## data augmentation
		if paraOI is not None:
			#print('use token to train')
			#paraOI = enrich_para(mu)
			#pdb.set_trace()
			iToken = self.TokenModel(paraOI).unsqueeze(1) #(batch_size, 1, 1024)
			make_up = torch.zeros(L-1, n_samples, self.rnn_hidden_size, dtype=torch.float32).to(self.device)
			h_t = torch.cat([iToken.squeeze(1).unsqueeze(0), make_up], dim=0)
			c_t = torch.zeros(L, n_samples, self.rnn_hidden_size, dtype=torch.float32).to(self.device)

			
		##

		else:
			h_t = torch.zeros(L, n_samples, self.rnn_hidden_size, dtype=torch.float32).to(self.device)
			c_t = torch.zeros(L, n_samples, self.rnn_hidden_size, dtype=torch.float32).to(self.device)
		#pdb.set_trace()
		# if mu is not None:
		# 	##
		# 	rnn_outputs = []
		# 	for i in range(x.shape[1]):
		# 		#pdb.set_trace()
		# 		enrich_input = torch.cat((x[:,i:i+1], iToken), dim = 1)
		# 		outputs, (h_t, c_t) = self.rnn(enrich_input, (h_t, c_t))
		# 		rnn_outputs.append(outputs[:,0:1])
		#
		# 	rnn_outputs=torch.cat(rnn_outputs, dim=1)
		# else:
		rnn_outputs, (hn, cn) = self.rnn(x, (h_t, c_t))
		## add a distribution args
		#pdb.set_trace()
		distr_args = self.distr_args(rnn_outputs=rnn_outputs)

		#pdb.set_trace()     
		log_prob = self.flow.log_prob(x[:,1:], distr_args[:,:-1])

		return -log_prob
	
	def distr_args(self, rnn_outputs):
		"""
		Returns the distribution of DeepVAR with respect to the RNN outputs.

		Parameters
		----------
		rnn_outputs
			Outputs of the unrolled RNN (batch_size, seq_len, num_cells)
		scale
			Mean scale for each time series (batch_size, 1, target_dim)

		Returns
		-------
		distr
			Distribution instance
		distr_args
			Distribution arguments
		"""
		(distr_args, ) = self.proj_dist_args(rnn_outputs)
		return distr_args






class CylinderTransformerParaPhysicsFlow(nn.Module):
	#(self, input_size, num_layers, hidden_size, num_cells, dropout_rate,n_blocks, n_hidden, mu_size, adj_var = 1, device = 'cuda:0', conditioning_length = 200)

	## let d_model = input_size
	
	def __init__(self, input_size, hidden_size, num_cells, dropout_rate, n_blocks, n_hidden, mu_size, adj_var = 1, device = 'cuda:0', conditioning_length = 200, context_length = 1, num_heads = 8, dim_feedforward_scale = 4, act_type = "gelu", num_encoder_layers = 3, num_decoder_layers = 3, mask_length = 400):
		super(CylinderTransformerParaPhysicsFlow, self).__init__()
		# [B, T, d_model] where d_model / num_heads is int
		print('\n')
		print(' using Cylinder Transformer')
		print('\n')
		d_model = input_size
		self.transformer = nn.Transformer(
			d_model=d_model,
			nhead=num_heads,
			num_encoder_layers=num_encoder_layers,
			num_decoder_layers=num_decoder_layers,
			dim_feedforward=dim_feedforward_scale * d_model,
			dropout=dropout_rate,
			activation=act_type,
		)

		#pdb.set_trace()
		self.flow = RealNVP(
			input_size=input_size,
			n_blocks=n_blocks,
			hidden_size=hidden_size,
			n_hidden = n_hidden,
			cond_label_size= 2*conditioning_length,
			adj_var = adj_var
		)
		#pdb.set_trace()
		#self.num_layers = num_layers
		self.rnn_hidden_size = num_cells
		self.flow_hidden_size = hidden_size
		self.device = device
		##
		self.target_dim = target_dim = input_size
		self.context_length = context_length
		if mu_size is not None:
			#MLPDense(config.paraEnrichDim, config.n_embd,[200, 200], True)
			self.TokenModel = MLPDense(6, input_size,[200, 200], True)
		
		self.distr_output = FlowOutput(
			self.flow, input_size=target_dim, cond_size=conditioning_length
		)
		#self.distr_output = 
		self.proj_dist_args = self.distr_output.get_args_proj(d_model)

		## transformer specific
		self.embed_dim = 1
		self.embed = nn.Embedding(
			num_embeddings=self.target_dim, embedding_dim=self.embed_dim
		)
		self.encoder_input = nn.Linear(input_size, d_model)
		self.decoder_input = nn.Linear(input_size, d_model)
		# mask
		self.register_buffer(
			"tgt_mask",
			self.transformer.generate_square_subsequent_mask(mask_length),
		)

		## init a positional enconding
		self.postionEncoding = PositionalEncoding(d_model = d_model)

	
	def forward(self, x, paraOI):
		return self.log_likelihood(x,paraOI)

	def sample(self,x, prediction_length, paraOI = None, sample_flag = True):
		'''
		:param x:torch.tensor(batch_size, 1, input_size)
		:param prediction_length:
		:return:
		'''
		#pdb.set_trace()
		## data augmentation
		if paraOI is not None:
			#paraOI = enrich_para(mu)
			iToken = self.TokenModel(paraOI).unsqueeze(1)
		print('sample model')

		outputs = []
		#L = self.num_layers
		n_samples = x.size(0)

		label_sec = x.size(1)-1

		if paraOI is not None:
			enc_inputs = torch.cat([iToken, x[:, :self.context_length, ...]], dim=1)
			


		else:
			enc_inputs = x[:, :self.context_length, ...]


		future_samples = []


		enc_out = self.transformer.encoder(
			self.encoder_input(enc_inputs).permute(1, 0, 2) # 100
		)



		for i in range(prediction_length):
				
			if i == 0:
				## x[1,2,3,4,5]
				## x[2,3,4,5]
				## 
				dec_input = x[:, 1:]

				
				## add position to dec_inputs
				dec_input_p = self.postionEncoding(dec_input.permute(1,0,2)).permute(1,0,2)

				tgt_mask = self.transformer.generate_square_subsequent_mask(dec_input.shape[1]).to('cuda:0')
				#[T,B,C]
				new_dec_output = self.transformer.decoder(self.decoder_input(dec_input_p).permute(1, 0, 2), enc_out, tgt_mask = tgt_mask)[-1].unsqueeze(0) #(1, batch_size, embd_length) need attention mask

				
			else:

				#print('i is', i)

				dec_input = torch.cat([dec_input, future_samples[-1]], dim=1)
				## add position to dec_inputs
				dec_input_p = self.postionEncoding(dec_input.permute(1,0,2)).permute(1,0,2)
				
				#print('shape of dec_input is', dec_input.shape)
				tgt_mask = self.transformer.generate_square_subsequent_mask(dec_input.shape[1]).to('cuda:0')
				#pdb.set_trace()
				# [T,B,C]
				new_dec_output = self.transformer.decoder(self.decoder_input(dec_input_p).permute(1, 0, 2), enc_out, tgt_mask = tgt_mask)[-1].unsqueeze(0)     #need attention mask


			### change new+_dec_output
			# [B,T,C]
			new_dec_output = new_dec_output.permute(1, 0, 2)
			#new_dec_output = torch.cat([new_dec_output, dec_input_p[:,-1,:].unsqueeze(1)], dim = 2)
			#pdb.set_trace()
			## [51,1,2048]
			distr_args = self.distr_args(decoder_output = new_dec_output)
			distr_args = torch.cat([distr_args, dec_input_p[:,-1,:].unsqueeze(1)], dim = 2)
			new_samples = self.flow.sample(cond = distr_args, sample_flag = sample_flag)
			future_samples.append(new_samples)

		future_samples = torch.cat(future_samples, dim=1)

		return future_samples


	def log_likelihood(self,x, paraOI):
		'''
		:param x: torch.tensor(batch_size,prediction_step,input_size)
		:param mu: torch.tensor(batch_size,1)
		:return:
		'''
		
		

		##
		## data augmentation
		if paraOI is not None:
			#print('generate token')
			## the paraOI was removed to the main code, since it requires global normalization
			#paraOI = enrich_para(mu)
			iToken = self.TokenModel(paraOI).unsqueeze(1) #(batch_size, 1, 1024)

			
		##
		n_samples = x.size(0)
		#L = self.num_layers
		self.context_length = 1
		if paraOI is not None:
			#print('cat token')
			enc_inputs = torch.cat([iToken,x[:, :self.context_length, ...]], dim=1)


		else:

			enc_inputs = x[:,:self.context_length,...] # 1
		dec_inputs = x[:,self.context_length:,...] # 400
		## add position to dec_inputs
		dec_iputs = self.postionEncoding(dec_inputs.permute(1,0,2)).permute(1,0,2)
		#print(dec_inputs.shape)

		enc_out = self.transformer.encoder(
			self.encoder_input(enc_inputs).permute(1, 0, 2) # 1
		)


		dec_output = self.transformer.decoder(
			self.decoder_input(dec_inputs).permute(1, 0, 2), # 400
			enc_out, # 1
			tgt_mask = self.tgt_mask # 400
		)


		## add a distribution args
		#pdb.set_trace()
		distr_args = self.distr_args(decoder_output=dec_output.permute(1,0,2))
		#[token,0]
		# decin:[1:n]
		# decout:[2:n+1]
		## [2:n]
		## [2',3',4',...n+1']
		## [1, 2, 3, 4,...n]
		## cat the correpsonding previous true label
		## (b,T,C)
		# [[1',2',3',...n'], [1, 2, 3, 4,...n]]

		## 1: [1']
		## 2: [2']
		## n: [n']
		## n+1 [1]
		## n+2 [2]
		distr_args = torch.cat([distr_args,dec_inputs], dim = 2)
		
		log_prob = self.flow.log_prob(x[:,2:], distr_args[:,:-1])

		return -log_prob
	
	def distr_args(self, decoder_output):
		"""
		Returns the distribution of DeepVAR with respect to the RNN outputs.

		Parameters
		----------
		rnn_outputs
			Outputs of the unrolled RNN (batch_size, seq_len, num_cells)
		scale
			Mean scale for each time series (batch_size, 1, target_dim)

		Returns
		-------
		distr
			Distribution instance
		distr_args
			Distribution arguments
		"""
		(distr_args, ) = self.proj_dist_args(decoder_output)
		return distr_args






class lsunTransformerParaPhysicsFlow(nn.Module):
	#(self, input_size, num_layers, hidden_size, num_cells, dropout_rate,n_blocks, n_hidden, mu_size, adj_var = 1, device = 'cuda:0', conditioning_length = 200)

	## let d_model = input_size
	
	def __init__(self, input_size, hidden_size, num_cells, dropout_rate, n_blocks, n_hidden, mu_size, adj_var = 1, device = 'cuda:0', conditioning_length = 200, context_length = 1, num_heads = 8, dim_feedforward_scale = 4, act_type = "gelu", num_encoder_layers = 3, num_decoder_layers = 3, mask_length = 400, d_model = None):
		super(lsunTransformerParaPhysicsFlow, self).__init__()
		# [B, T, d_model] where d_model / num_heads is int
		#d_model = input_size
		print('d_model is', d_model)
		self.transformer = nn.Transformer(
			d_model=d_model,
			nhead=num_heads,
			num_encoder_layers=num_encoder_layers,
			num_decoder_layers=num_decoder_layers,
			dim_feedforward=dim_feedforward_scale * d_model,
			dropout=dropout_rate,
			activation=act_type,
		)

		#pdb.set_trace()
		self.flow = RealNVP(
			input_size=input_size,
			n_blocks=n_blocks,
			hidden_size=hidden_size,
			n_hidden = n_hidden,
			cond_label_size=conditioning_length,
			adj_var = adj_var
		)
		#pdb.set_trace()
		#self.num_layers = num_layers
		self.rnn_hidden_size = num_cells
		self.flow_hidden_size = hidden_size
		self.device = device
		##
		self.target_dim = target_dim = input_size
		self.context_length = context_length
		if mu_size is not None:
			#MLPDense(config.paraEnrichDim, config.n_embd,[200, 200], True)
			self.TokenModel = MLPDense(6, input_size,[200, 200], True)
		
		self.distr_output = FlowOutput(
			self.flow, input_size=target_dim, cond_size=conditioning_length
		)
		#self.distr_output = 
		self.proj_dist_args = self.distr_output.get_args_proj(d_model)

		## transformer specific
		self.embed_dim = 1
		self.embed = nn.Embedding(
			num_embeddings=self.target_dim, embedding_dim=self.embed_dim
		)
		self.encoder_input = nn.Linear(input_size, d_model)
		self.decoder_input = nn.Linear(input_size, d_model)
		# mask
		self.register_buffer(
			"tgt_mask",
			self.transformer.generate_square_subsequent_mask(mask_length),
		)

		##

	
	def forward(self, x, paraOI):
		return self.log_likelihood(x,paraOI)

	def sample(self,x, prediction_length, paraOI = None, sample_flag = True):
		'''
		:param x:torch.tensor(batch_size, 1, input_size)
		:param prediction_length:
		:return:
		'''
		#pdb.set_trace()
		## data augmentation
		if paraOI is not None:
			#paraOI = enrich_para(mu)
			iToken = self.TokenModel(paraOI).unsqueeze(1)
		print('sample model')

		outputs = []
		#L = self.num_layers
		n_samples = x.size(0)

		label_sec = x.size(1)-1

		if paraOI is not None:
			enc_inputs = torch.cat([iToken, x[:, :self.context_length, ...]], dim=1)
			


		else:
			enc_inputs = x[:, :self.context_length, ...]


		future_samples = []


		enc_out = self.transformer.encoder(
			self.encoder_input(enc_inputs).permute(1, 0, 2) # 100
		)

		

		for i in range(prediction_length):
				
			if i == 0:
				## x[1,2,3,4,5]
				## x[2,3,4,5]
				## 
				dec_input = x[:, 1:]
				tgt_mask = self.transformer.generate_square_subsequent_mask(dec_input.shape[1]).to('cuda:0')

				new_dec_output = self.transformer.decoder(self.decoder_input(dec_input).permute(1, 0, 2), enc_out, tgt_mask = tgt_mask)[-1].unsqueeze(0) #(batch_size,1 embd_length) need attention mask
			else:

				#print('i is', i)

				dec_input = torch.cat([dec_input, future_samples[-1]], dim=1)
				#print('shape of dec_input is', dec_input.shape)
				tgt_mask = self.transformer.generate_square_subsequent_mask(dec_input.shape[1]).to('cuda:0')

				new_dec_output = self.transformer.decoder(self.decoder_input(dec_input).permute(1, 0, 2), enc_out, tgt_mask = tgt_mask)[-1].unsqueeze(0)     #need attention mask
			distr_args = self.distr_args(decoder_output = new_dec_output.permute(1, 0, 2))
			new_samples = self.flow.sample(cond = distr_args, sample_flag = sample_flag)
			future_samples.append(new_samples)

		future_samples = torch.cat(future_samples, dim=1)

		return future_samples


	def log_likelihood(self,x, paraOI):
		'''
		:param x: torch.tensor(batch_size,prediction_step,input_size)
		:param mu: torch.tensor(batch_size,1)
		:return:
		'''
		#pdb.set_trace()

		## data augmentation
		if paraOI is not None:
			#print('generate token')
			## the paraOI was removed to the main code, since it requires global normalization
			#paraOI = enrich_para(mu)
			iToken = self.TokenModel(paraOI).unsqueeze(1) #(batch_size, 1, 1024)

			
		##
		n_samples = x.size(0)
		#L = self.num_layers
		self.context_length = 1
		if paraOI is not None:
			#print('cat token')
			enc_inputs = torch.cat([iToken,x[:, :self.context_length, ...]], dim=1)


		else:
			#print('no Token')
			enc_inputs = x[:,:self.context_length,...] # 1
		dec_inputs = x[:,self.context_length:,...] # 400

		enc_out = self.transformer.encoder(
			self.encoder_input(enc_inputs).permute(1, 0, 2) # 1
		)


		dec_output = self.transformer.decoder(
			self.decoder_input(dec_inputs).permute(1, 0, 2), # 400
			enc_out, # 1
			tgt_mask = self.tgt_mask # 400
		)

		## add a distribution args
		#pdb.set_trace()
		distr_args = self.distr_args(decoder_output=dec_output.permute(1,0,2))


		_hidden = x[:,2:].cpu().detach().numpy()
		_condition = distr_args[:,:-1].cpu().detach().numpy()
		_hidden = _hidden[:5]
		_condition = _condition[:5]
		_hidden = np.swapaxes(_hidden,1,2)
		_condition = np.swapaxes(_condition,1,2)
		_hidden = _hidden[..., None]
		_condition = _condition[..., None]
		np.save('train_hidden_', _hidden)
		np.save('train_condition_', _condition)
		pdb.set_trace()
		log_prob = self.flow.log_prob(x[:,2:], distr_args[:,:-1])

		return -log_prob
	
	def distr_args(self, decoder_output):
		"""
		Returns the distribution of DeepVAR with respect to the RNN outputs.

		Parameters
		----------
		rnn_outputs
			Outputs of the unrolled RNN (batch_size, seq_len, num_cells)
		scale
			Mean scale for each time series (batch_size, 1, target_dim)

		Returns
		-------
		distr
			Distribution instance
		distr_args
			Distribution arguments
		"""
		(distr_args, ) = self.proj_dist_args(decoder_output)
		return distr_args







class lsunParaSinePhysicsFlow(nn.Module):
	def __init__(self, input_size, num_layers, hidden_size, num_cells, dropout_rate,n_blocks, n_hidden, mu_size, adj_var = 1, device = 'cuda:0'):
		super(lsunParaSinePhysicsFlow, self).__init__()
		
		self.rnn = nn.LSTM(
			input_size=input_size,
			hidden_size=num_cells,
			num_layers=num_layers,
			dropout=dropout_rate,
			batch_first=True,
		)
		self.linear = nn.Linear(num_cells, input_size)

		self.flow = RealNVP(
			input_size=input_size,
			n_blocks=n_blocks,
			hidden_size=hidden_size,
			n_hidden = n_hidden,
			cond_label_size=input_size,
			adj_var = adj_var
		)
		#pdb.set_trace()
		self.num_layers = num_layers
		self.hidden_size = hidden_size
		self.device = device

	
	def forward(self, x, mu = None):
		return self.log_likelihood(x,mu)

	def sample(self,x, prediction_length,sample_flag = True):
		'''
		:param x:torch.tensor(batch_size, 1, input_size)
		:param prediction_length:
		:return:
		'''
		outputs = []
		L = self.num_layers
		n_samples = x.size(0)
		h_t = torch.zeros(L, n_samples, self.hidden_size, dtype=torch.float32).to(self.device)
		c_t = torch.zeros(L, n_samples, self.hidden_size, dtype=torch.float32).to(self.device)
		label_sec = x.size(1)-1
		for i in range(label_sec):
			if i==0:
				output, (h_t,c_t) = self.rnn(x[:,0:1,:], (h_t, c_t))
				
			elif i>0:
				output, (h_t,c_t) = self.rnn(x[:,i:i+1,:], (h_t, c_t))
			### linear layer
			output = self.linear(output)
			###
			new_samples = self.flow.sample(cond=output)
			outputs.append(new_samples)
		
		future_samples = []
		for i in range(prediction_length):
			output, (h_t, c_t) = self.rnn(new_samples, (h_t, c_t))
			output = self.linear(output)
			new_samples = self.flow.sample(cond=output)
			future_samples.append(new_samples)
		future_samples = torch.cat(future_samples, dim=1)

		return future_samples


	def log_likelihood(self,x, mu):
		'''
		:param x: torch.tensor(batch_size,prediction_step,input_size)
		:param mu: torch.tensor(batch_size,1)
		:return:
		'''
		n_samples = x.size(0)
		L = self.num_layers

		h_t = torch.zeros(L, n_samples, self.hidden_size, dtype=torch.float32).to(self.device)
		c_t = torch.zeros(L, n_samples, self.hidden_size, dtype=torch.float32).to(self.device)

		
		rnn_outputs, (hn, cn) = self.rnn(x, (h_t, c_t))   
		rnn_outputs = self.linear(rnn_outputs)
		log_prob = self.flow.log_prob(x[:,1:], rnn_outputs[:,:-1])

		return -log_prob




#This one works for LSTM.
class lsunParaPhysicsFlow(nn.Module):
	def __init__(self, input_size, num_layers, hidden_size, num_cells, dropout_rate,n_blocks, n_hidden, mu_size, adj_var = 1, device = 'cuda:0', conditioning_length = 200):
		super(lsunParaPhysicsFlow, self).__init__()
		print('using rnn no token')
		time.sleep(1)
		self.rnn = nn.LSTM(
			input_size=input_size,
			hidden_size=num_cells,
			num_layers=num_layers,
			dropout=dropout_rate,
			batch_first=True,
		)
		#pdb.set_trace()
		self.flow = RealNVP(
			input_size=input_size,
			n_blocks=n_blocks,
			hidden_size=hidden_size,
			n_hidden = n_hidden,
			cond_label_size=conditioning_length,
			adj_var = adj_var
		)
		#pdb.set_trace()
		self.num_layers = num_layers
		self.rnn_hidden_size = num_cells
		self.flow_hidden_size = hidden_size
		self.device = device

		target_dim = input_size
		if mu_size is not None:
			#MLPDense(config.paraEnrichDim, config.n_embd,[200, 200], True)
			self.TokenModel = MLPDense(6, num_cells,[200, 200], True)
		
		self.distr_output = FlowOutput(
			self.flow, input_size=target_dim, cond_size=conditioning_length
		)
		self.proj_dist_args = self.distr_output.get_args_proj(num_cells)
		#self.proj_dist_args = self.distr_output.get_args_proj(input_size)

	
	def forward(self, x, mu = None):
		return self.log_likelihood(x,mu)

	def sample(self,x, prediction_length, paraOI = None, sample_flag = True):
		'''
		:param x:torch.tensor(batch_size, 1, input_size)
		:param prediction_length:
		:return:
		'''
		## data augmentation
		#if mu is not None:
			#paraOI = enrich_para(mu)
			#iToken = self.TokenModel(paraOI).unsqueeze(1)
		print('sample model')
		#pdb.set_trace()
		outputs = []
		L = self.num_layers
		n_samples = x.size(0)
		h_t = torch.zeros(L, n_samples, self.rnn_hidden_size, dtype=torch.float32).to(self.device)
		c_t = torch.zeros(L, n_samples, self.rnn_hidden_size, dtype=torch.float32).to(self.device)
		label_sec = x.size(1)-1

		if paraOI is not None:
			
			rnn_outputs = []
			for i in range(label_sec):
				#if i==0:
					#enrich_input = torch.cat((x[:,i:i+1], iToken), dim = 1)
					#output, (h_t, c_t) = self.rnn(enrich_input, (h_t, c_t))
					
			   # elif i>0:
			   ## give enrich data to rnn
				#pdb.set_trace()
				enrich_input = torch.cat((x[:,i:i+1], iToken), dim = 1)
				output, (h_t, c_t) = self.rnn(enrich_input, (h_t, c_t))
				
				new_samples = self.flow.sample(cond=output[:,0:1], sample_flag = sample_flag)
				outputs.append(new_samples)
			
			future_samples = []
			for i in range(prediction_length):
				## give enrich data to rnn
				enrich_input = torch.cat((new_samples, iToken), dim = 1)
				output, (h_t, c_t) = self.rnn(enrich_input, (h_t, c_t))

				new_samples = self.flow.sample(cond=output[:,0:1])
				future_samples.append(new_samples)
			future_samples = torch.cat(future_samples, dim=1)

		else:

			# for i in range(label_sec):
			#     if i==0:
			#         output, (h_t,c_t) = self.rnn(x[:,0:1,:], (h_t, c_t))
			#     elif i>0:
			#         output, (h_t,c_t) = self.rnn(x[:,i:i+1,:], (h_t, c_t))
			#     #pdb.set_trace()
			#     #new_samples = self.flow.sample(cond=output)
			#     #outputs.append(new_samples)
			#rnn_outputs, (h_t, c_t) = self.rnn(x, (h_t, c_t))
			#output = rnn_outputs[:,-1].unsqueeze(1)
			
			#pdb.set_trace()
			future_samples = []
			for i in range(prediction_length):
				if i == 0:
					#output, (h_t, c_t) = self.rnn(x[:,-1].unsqueeze(1), (h_t, c_t))
					rnn_outputs, (h_t, c_t) = self.rnn(x, (h_t, c_t))
					output = rnn_outputs[:,-1].unsqueeze(1)
				else:
					
					output, (h_t, c_t) = self.rnn(new_samples, (h_t, c_t))
					#x = torch.cat((x,future_samples[-1]), dim = 1)
					#rnn_outputs, (h_t, c_t) = self.rnn(x, (h_t, c_t))
					#output = rnn_outputs[:,-1].unsqueeze(1)

				distr_args = self.distr_args(rnn_outputs = output)
				new_samples = self.flow.sample(cond = distr_args, sample_flag = sample_flag)
				future_samples.append(new_samples)

			future_samples = torch.cat(future_samples, dim=1)

		return future_samples


	def log_likelihood(self,x, mu):
		'''
		:param x: torch.tensor(batch_size,prediction_step,input_size)
		:param mu: torch.tensor(batch_size,1)
		:return:
		'''
		#pdb.set_trace()

		## data augmentation
		if mu is not None:
			paraOI = enrich_para(mu)
			iToken = self.TokenModel(paraOI).unsqueeze(1)
			
		##
		n_samples = x.size(0)
		L = self.num_layers

		h_t = torch.zeros(L, n_samples, self.rnn_hidden_size, dtype=torch.float32).to(self.device)
		c_t = torch.zeros(L, n_samples, self.rnn_hidden_size, dtype=torch.float32).to(self.device)
		#pdb.set_trace()
		if mu is not None:
			## 
			rnn_outputs = []
			for i in range(x.shape[1]):
				#pdb.set_trace()
				enrich_input = torch.cat((x[:,i:i+1], iToken), dim = 1)
				outputs, (h_t, c_t) = self.rnn(enrich_input, (h_t, c_t))
				rnn_outputs.append(outputs[:,0:1])

			rnn_outputs=torch.cat(rnn_outputs, dim=1)
		else:
			rnn_outputs, (hn, cn) = self.rnn(x, (h_t, c_t))
			## add a distribution args
			distr_args = self.distr_args(rnn_outputs=rnn_outputs)
			#pdb.set_trace()
		#pdb.set_trace()     
		log_prob = self.flow.log_prob(x[:,1:], distr_args[:,:-1])

		return -log_prob
	
	def distr_args(self, rnn_outputs):
		"""
		Returns the distribution of DeepVAR with respect to the RNN outputs.

		Parameters
		----------
		rnn_outputs
			Outputs of the unrolled RNN (batch_size, seq_len, num_cells)
		scale
			Mean scale for each time series (batch_size, 1, target_dim)

		Returns
		-------
		distr
			Distribution instance
		distr_args
			Distribution arguments
		"""
		(distr_args, ) = self.proj_dist_args(rnn_outputs)
		return distr_args


#old one This one works for LSTM.
class oldlsunParaPhysicsFlow(nn.Module):
	def __init__(self, input_size, num_layers, hidden_size, num_cells, dropout_rate,n_blocks, n_hidden, mu_size, adj_var = 1, device = 'cuda:0', conditioning_length = 200):
		super(lsunParaPhysicsFlow, self).__init__()
		print('using rnn no token')
		time.sleep(1)
		self.rnn = nn.LSTM(
			input_size=input_size,
			hidden_size=num_cells,
			num_layers=num_layers,
			dropout=dropout_rate,
			batch_first=True,
		)
		#pdb.set_trace()
		self.flow = RealNVP(
			input_size=input_size,
			n_blocks=n_blocks,
			hidden_size=hidden_size,
			n_hidden = n_hidden,
			cond_label_size=conditioning_length,
			adj_var = adj_var
		)
		#pdb.set_trace()
		self.num_layers = num_layers
		self.rnn_hidden_size = num_cells
		self.flow_hidden_size = hidden_size
		self.device = device

		target_dim = input_size
		if mu_size is not None:
			#MLPDense(config.paraEnrichDim, config.n_embd,[200, 200], True)
			self.TokenModel = MLPDense(6, num_cells,[200, 200], True)
		
		self.distr_output = FlowOutput(
			self.flow, input_size=target_dim, cond_size=conditioning_length
		)
		self.proj_dist_args = self.distr_output.get_args_proj(num_cells)
		#self.proj_dist_args = self.distr_output.get_args_proj(input_size)

	
	def forward(self, x, mu = None):
		return self.log_likelihood(x,mu)

	def sample(self,x, prediction_length, paraOI = None, sample_flag = True):
		'''
		:param x:torch.tensor(batch_size, 1, input_size)
		:param prediction_length:
		:return:
		'''
		## data augmentation
		#if mu is not None:
			#paraOI = enrich_para(mu)
			#iToken = self.TokenModel(paraOI).unsqueeze(1)
		print('sample model')
		#pdb.set_trace()
		outputs = []
		L = self.num_layers
		n_samples = x.size(0)
		h_t = torch.zeros(L, n_samples, self.rnn_hidden_size, dtype=torch.float32).to(self.device)
		c_t = torch.zeros(L, n_samples, self.rnn_hidden_size, dtype=torch.float32).to(self.device)
		label_sec = x.size(1)-1

		if paraOI is not None:
			
			rnn_outputs = []
			for i in range(label_sec):
				#if i==0:
					#enrich_input = torch.cat((x[:,i:i+1], iToken), dim = 1)
					#output, (h_t, c_t) = self.rnn(enrich_input, (h_t, c_t))
					
			   # elif i>0:
			   ## give enrich data to rnn
				#pdb.set_trace()
				enrich_input = torch.cat((x[:,i:i+1], iToken), dim = 1)
				output, (h_t, c_t) = self.rnn(enrich_input, (h_t, c_t))
				
				new_samples = self.flow.sample(cond=output[:,0:1], sample_flag = sample_flag)
				outputs.append(new_samples)
			
			future_samples = []
			for i in range(prediction_length):
				## give enrich data to rnn
				enrich_input = torch.cat((new_samples, iToken), dim = 1)
				output, (h_t, c_t) = self.rnn(enrich_input, (h_t, c_t))

				new_samples = self.flow.sample(cond=output[:,0:1])
				future_samples.append(new_samples)
			future_samples = torch.cat(future_samples, dim=1)

		else:

			# for i in range(label_sec):
			#     if i==0:
			#         output, (h_t,c_t) = self.rnn(x[:,0:1,:], (h_t, c_t))
			#     elif i>0:
			#         output, (h_t,c_t) = self.rnn(x[:,i:i+1,:], (h_t, c_t))
			#     #pdb.set_trace()
			#     #new_samples = self.flow.sample(cond=output)
			#     #outputs.append(new_samples)
			rnn_outputs, (h_t, c_t) = self.rnn(x, (h_t, c_t))
			output = rnn_outputs[:,-1].unsqueeze(1)
			
			#pdb.set_trace()
			future_samples = []
			for i in range(prediction_length):
				if i == 0:
					output, (h_t, c_t) = self.rnn(x[:,-1].unsqueeze(1), (h_t, c_t))
				else:
					output, (h_t, c_t) = self.rnn(new_samples, (h_t, c_t))

				distr_args = self.distr_args(rnn_outputs = output)
				new_samples = self.flow.sample(cond = distr_args, sample_flag = sample_flag)
				future_samples.append(new_samples)

			future_samples = torch.cat(future_samples, dim=1)

		return future_samples


	def log_likelihood(self,x, mu):
		'''
		:param x: torch.tensor(batch_size,prediction_step,input_size)
		:param mu: torch.tensor(batch_size,1)
		:return:
		'''
		#pdb.set_trace()

		## data augmentation
		if mu is not None:
			paraOI = enrich_para(mu)
			iToken = self.TokenModel(paraOI).unsqueeze(1)
			
		##
		n_samples = x.size(0)
		L = self.num_layers

		h_t = torch.zeros(L, n_samples, self.rnn_hidden_size, dtype=torch.float32).to(self.device)
		c_t = torch.zeros(L, n_samples, self.rnn_hidden_size, dtype=torch.float32).to(self.device)
		#pdb.set_trace()
		if mu is not None:
			## 
			rnn_outputs = []
			for i in range(x.shape[1]):
				#pdb.set_trace()
				enrich_input = torch.cat((x[:,i:i+1], iToken), dim = 1)
				outputs, (h_t, c_t) = self.rnn(enrich_input, (h_t, c_t))
				rnn_outputs.append(outputs[:,0:1])

			rnn_outputs=torch.cat(rnn_outputs, dim=1)
		else:
			rnn_outputs, (hn, cn) = self.rnn(x, (h_t, c_t))
			## add a distribution args
			distr_args = self.distr_args(rnn_outputs=rnn_outputs)
			#pdb.set_trace()
		#pdb.set_trace()     
		log_prob = self.flow.log_prob(x[:,1:], distr_args[:,:-1])

		return -log_prob
	
	def distr_args(self, rnn_outputs):
		"""
		Returns the distribution of DeepVAR with respect to the RNN outputs.

		Parameters
		----------
		rnn_outputs
			Outputs of the unrolled RNN (batch_size, seq_len, num_cells)
		scale
			Mean scale for each time series (batch_size, 1, target_dim)

		Returns
		-------
		distr
			Distribution instance
		distr_args
			Distribution arguments
		"""
		(distr_args, ) = self.proj_dist_args(rnn_outputs)
		return distr_args


class xu_transformer(nn.Module):
	def __init__(self, input_size, num_layers, hidden_size, num_cells, dropout_rate, n_blocks, n_hidden, mu_size,
				 adj_var=1, device='cuda:0', conditioning_length=200):
		super(xu_transformer, self).__init__()
		#decoder_layer = nn.TransformerDecoderLayer(d_model=input_size, nhead=8)
		## ctx length?? why need to be seq length now?
		# for cylinder 402
		self.transformer = AttentionDecoder(n_embd=input_size, embd_pdrop=0.0,n_ctx=251,n_layer=1,layer_norm_epsilon=1e-5,attn_pdrop=0.0,
											resid_pdrop=0.0,n_head=4, activation_function='relu', position=0)  #should be added to args

		self.flow = RealNVP(
			input_size=input_size,
			n_blocks=n_blocks,
			hidden_size=hidden_size,
			n_hidden=n_hidden,
			cond_label_size=conditioning_length,
			adj_var=adj_var
		)
		# pdb.set_trace()
		self.num_layers = num_layers
		self.rnn_hidden_size = num_cells
		self.flow_hidden_size = hidden_size
		self.device = device

		target_dim = input_size
		if mu_size is not None:
			# MLPDense(config.paraEnrichDim, config.n_embd,[200, 200], True)
			#self.TokenModel = MLPDense(6, num_cells, [200, 200], True)
			self.TokenModel = MLPDense(6, input_size, [200, 200], True)
			

		self.distr_output = FlowOutput(
			self.flow, input_size=target_dim, cond_size=conditioning_length
		)
		#self.proj_dist_args = self.distr_output.get_args_proj(num_cells)
		self.proj_dist_args = self.distr_output.get_args_proj(input_size)
	def forward(self, x, mu=None):
		return self.log_likelihood(x, mu)

	def sample(self, x, prediction_length, paraOI=None,sample_flag = True):
		'''
		:param x:torch.tensor(batch_size, 1, input_size)
		:param prediction_length:
		:return:
		'''
		if paraOI is not None:
			print('sample generate token')
			#paraOI = enrich_para(mu)
			iToken = self.TokenModel(paraOI).unsqueeze(1) #(batch_size, 1, 1024)
		
		if paraOI is not None:
			print('sample cat token')
			x = torch.cat([iToken,x], dim=1)

		#### post processing for only first and last
		#pdb.set_trace()
		####

		future_samples = []

		temp_list = []


		for i in range(prediction_length):
				
			if i == 0:
				#[b,Tpast,C]
				dec_input = x#.unsqueeze(1)  #? -1 -> 1
				tgt_mask = None
				#new_dec_output = self.transformer.decoder(self.decoder_input(dec_input).permute(1, 0, 2), enc_out, tgt_mask = tgt_mask)[-1].unsqueeze(0) #(batch_size,1 embd_length) need attention mask
				znp1,past,_,attention = self.transformer(dec_input, attention_mask=tgt_mask) #(batch_size,1 embd_length) need attention mask
				new_dec_output = znp1[:,-1].unsqueeze(1)

				#attention = new_dec_output_list[-1]
				#pdb.set_trace()
				attention = attention[0][0,:,-1,:]
			else:

				#print('i is', i)

				#dec_input = torch.cat([dec_input, future_samples[-1]], dim=1)
				dec_input = future_samples[-1]
				#print('shape of dec_input is', dec_input.shape)
				#tgt_mask = None
				#new_dec_output_list = self.transformer(dec_input, attention_mask=tgt_mask) #(batch_size,1 embd_length) need attention mask
				znp1,past,_,attention = self.transformer(dec_input, past = past) #(batch_size,1 embd_length) need attention mask
				new_dec_output = znp1[:,-1].unsqueeze(1)
				#attention = new_dec_output_list[-1]
				#pdb.set_trace()
				attention = attention[0][0,:,-1,:]
				#new_dec_output = self.transformer.decoder(self.decoder_input(dec_input).permute(1, 0, 2), enc_out, tgt_mask = tgt_mask)[-1].unsqueeze(0)     #need attention mask
			### 2/3 attention blocks, distr_args 1024
			distr_args = self.distr_args(decoder_output = new_dec_output)
			new_samples = self.flow.sample(cond = distr_args, sample_flag = sample_flag)
			future_samples.append(new_samples)
			#temp_list.append(attention)
		#pdb.set_trace()
		future_samples = torch.cat(future_samples, dim=1)

		return future_samples#, temp_list

	def log_likelihood(self, x, paraOI):
		'''
		:param x: torch.tensor(batch_size,prediction_step,input_size)
		:param mu: torch.tensor(batch_size,1)
		:return:
		'''
		# pdb.set_trace()
		
		### modified 

		## data augmentation
		if paraOI is not None:
			#print('generate token')
			#paraOI = enrich_para(mu)
			iToken = self.TokenModel(paraOI).unsqueeze(1) #(batch_size, 1, 1024)
			#pdb.set_trace()
		if paraOI is not None:
			#print('cat token')
			x = torch.cat([iToken,x], dim=1)

		####

		# We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
		#mask = torch.tril(torch.ones((batch_size, n, n)))
		#mask = torch.ones([batch_size, 1, 1, x.shape[1]]).to('cuda:0')
		mask = None

		#transformer_outputs, _  = self.transformer(x, attention_mask=mask)

		transformer_outputs_list  = self.transformer(x, attention_mask=mask)
		transformer_outputs = transformer_outputs_list[0]
		#print('stop at the transformer output')
		#print('len of transformer outputs is', len(transformer_outputs_list))
		#pdb.set_trace()
		## add a distribution args
		distr_args = self.distr_args(decoder_output=transformer_outputs)
		# pdb.set_trace()
		#pdb.set_trace()

		#[token,0-250]
		# [t',0-250]

		## x:[0:250]
		## distrargs[1:251]
		## x[1:] (1:250)
		## distr_args[:-1] (1:250)
		log_prob = self.flow.log_prob(x[:, 2:], distr_args[:, 1:-1])

		return -log_prob

	def distr_args(self, decoder_output):
		"""
		Returns the distribution of DeepVAR with respect to the RNN outputs.
		Parameters
		----------
		rnn_outputs
			Outputs of the unrolled RNN (batch_size, seq_len, num_cells)
		scale
			Mean scale for each time series (batch_size, 1, target_dim)
		Returns
		-------
		distr
			Distribution instance
		distr_args
			Distribution arguments
		"""
		(distr_args,) = self.proj_dist_args(decoder_output)
		return distr_args


class lsunPhysicsFlow(nn.Module):
	def __init__(self, input_size, num_layers, hidden_size, num_cells, dropout_rate,n_blocks, n_hidden, mu_size, device = 'cuda:0'):
		super(lsunPhysicsFlow, self).__init__()
		
		self.rnn = nn.LSTM(
			input_size=input_size,
			hidden_size=num_cells,
			num_layers=num_layers,
			dropout=dropout_rate,
			batch_first=True,
		)

		self.flow = RealNVP(
			input_size=input_size,
			n_blocks=n_blocks,
			hidden_size=hidden_size,
			n_hidden = n_hidden,
			cond_label_size=input_size,
		)
		#pdb.set_trace()
		self.num_layers = num_layers
		self.hidden_size = hidden_size
		self.device = device

	
	def forward(self, x, mu = None):
		return self.log_likelihood(x,mu)

	def sample(self,x, prediction_length):
		'''
		:param x:torch.tensor(batch_size, 1, input_size)
		:param prediction_length:
		:return:
		'''
		outputs = []
		L = self.num_layers
		n_samples = x.size(0)
		h_t = torch.zeros(L, n_samples, self.hidden_size, dtype=torch.float32).to(self.device)
		c_t = torch.zeros(L, n_samples, self.hidden_size, dtype=torch.float32).to(self.device)
		label_sec = x.size(1)-1
		for i in range(label_sec):
			pdb.set_trace()
			if i==0:
				output, (h_t,c_t) = self.rnn(x[:,0:1,:], (h_t, c_t))
			elif i>0:
				output, (h_t,c_t) = self.rnn(x[:,i:i+1,:], (h_t, c_t))
			new_samples = self.flow.sample(cond=output)
			outputs.append(new_samples)
		
		future_samples = []
		for i in range(prediction_length):
			output, (h_t, c_t) = self.rnn(new_samples, (h_t, c_t))
			new_samples = self.flow.sample(cond=output)
			future_samples.append(new_samples)
		future_samples = torch.cat(future_samples, dim=1)

		return future_samples


	def log_likelihood(self,x, mu):
		'''
		:param x: torch.tensor(batch_size,prediction_step,input_size)
		:param mu: torch.tensor(batch_size,1)
		:return:
		'''
		n_samples = x.size(0)
		L = self.num_layers

		h_t = torch.zeros(L, n_samples, self.hidden_size, dtype=torch.float32).to(self.device)
		c_t = torch.zeros(L, n_samples, self.hidden_size, dtype=torch.float32).to(self.device)

		
		rnn_outputs, (hn, cn) = self.rnn(x, (h_t, c_t))    
		log_prob = self.flow.log_prob(x[:,1:], rnn_outputs[:,:-1])

		return -log_prob


class PhysicsFlow(nn.Module):
	def __init__(self, input_size, num_layers, hidden_size, num_cells, dropout_rate,n_blocks, n_hidden, mu_size):
		super().__init__()

		self.rnn = nn.LSTM(
			input_size=input_size,
			hidden_size=num_cells,
			num_layers=num_layers,
			dropout=dropout_rate,
			batch_first=True,
		)

		self.flow = RealNVP(
			input_size=input_size,
			n_blocks=n_blocks,
			hidden_size=hidden_size,
			n_hidden = n_hidden,
			cond_label_size=input_size,
		)

		#self.prediction_step=prediction_step




	def forward(self, x):
		return self.log_likelihood(x,None)



	def sample(self,x, prediction_length):
		'''
		:param x:torch.tensor(batch_size, 1, input_size)
		:param prediction_length:
		:return:
		'''
		future_samples = []
		state=None
		new_samples=x
		for i in range(prediction_length):
			outputs,state=self.rnn(new_samples, state)
			new_samples = self.flow.sample(cond=outputs[:,-1].unsqueeze(1))
			future_samples.append(new_samples)
		x=torch.cat(future_samples, dim=1)
		return x









	def log_likelihood(self,x, mu):
		'''
		:param x: torch.tensor(batch_size,prediction_step,input_size)
		:param mu: torch.tensor(batch_size,1)
		:return:
		'''

		rnn_outputs, states= self.rnn(x)
		log_prob = self.flow.log_prob(x[:,1:], rnn_outputs[:,:-1])

		return -log_prob



class PhysicsFlow_plus_linear(nn.Module):
	def __init__(self, input_size, num_layers, hidden_size, num_cells, dropout_rate,n_blocks, n_hidden, prediction_step, mu_size):
		super().__init__()

		self.rnn = nn.LSTM(
			input_size=input_size,
			hidden_size=num_cells,
			num_layers=num_layers,
			dropout=dropout_rate,
			batch_first=True,
		)

		self.flow = RealNVP(
			input_size=input_size,
			n_blocks=n_blocks,
			hidden_size=hidden_size,
			n_hidden = n_hidden,
			cond_label_size=input_size,
		)

		self.prediction_step=prediction_step
		self.time_encoder=MLP(nIn=input_size,nOut=input_size,Hidlayer=[input_size], withReLU=True)




	def forward(self, x):
		return self.log_likelihood(x,None)




	def sample(self,x, prediction_length):
		'''
		:param x:torch.tensor(batch_size, 1, input_size)
		:param prediction_length:
		:return:
		'''
		future_samples = []
		state=None
		new_samples=x
		for i in range(prediction_length):
			outputs,state=self.rnn(new_samples, state)
			new_samples = self.flow.sample(cond=outputs[:,-1].unsqueeze(1))
			future_samples.append(new_samples)
		x=torch.cat(future_samples, dim=1)
		return x









	def log_likelihood(self,x, mu):
		'''
		:param x: torch.tensor(batch_size,prediction_step,input_size)
		:param mu: torch.tensor(batch_size,1)
		:return:
		'''

		rnn_outputs, states= self.rnn(x)
		rnn_outputs = self.time_encoder(rnn_outputs)
		log_prob = self.flow.log_prob(x[:,1:], rnn_outputs[:,:-1])

		return -log_prob




class PhysicsFlow_residual(nn.Module):
	def __init__(self, input_size, num_layers, hidden_size, num_cells, dropout_rate,n_blocks, n_hidden, mu_size):
		super().__init__()

		self.rnn = nn.LSTM(
			input_size=input_size,
			hidden_size=num_cells,
			num_layers=num_layers,
			dropout=dropout_rate,
			batch_first=True,
		)

		self.flow = RealNVP(
			input_size=input_size,
			n_blocks=n_blocks,
			hidden_size=hidden_size,
			n_hidden = n_hidden,
			cond_label_size=input_size,
		)

	   # self.prediction_step=prediction_step




	def forward(self, x):
		return self.log_likelihood(x,None)



	def sample(self,x, prediction_length):
		'''
		:param x:torch.tensor(batch_size, 1, input_size)
		:param prediction_length:
		:return:
		'''
		future_samples = []
		state=None
		new_samples=x
		for i in range(prediction_length):
			outputs,state=self.rnn(new_samples, state)
			sample_residual = self.flow.sample(cond=outputs[:,-1].unsqueeze(1))

			new_samples = new_samples + sample_residual

			future_samples.append(new_samples)
		x=torch.cat(future_samples, dim=1)
		return x









	def log_likelihood(self,x, mu):
		'''
		:param x: torch.tensor(batch_size,prediction_step,input_size)
		:param mu: torch.tensor(batch_size,1)
		:return:
		'''

		rnn_outputs, states= self.rnn(x)
		residual = x[:,1:]-x[:,:-1]

		log_prob = self.flow.log_prob(residual, rnn_outputs[:,:-1])

		return -log_prob


class PhysicsFlow_initial_zero(nn.Module):
	def __init__(self, input_size, num_layers, hidden_size, num_cells, dropout_rate,n_blocks, n_hidden, mu_size):
		super().__init__()

		self.rnn = nn.LSTM(
			input_size=input_size,
			hidden_size=num_cells,
			num_layers=num_layers,
			dropout=dropout_rate,
			batch_first=True,
		)

		self.flow = RealNVP(
			input_size=input_size,
			n_blocks=n_blocks,
			hidden_size=hidden_size,
			n_hidden = n_hidden,
			cond_label_size=input_size,
		)

		#self.prediction_step=prediction_step




	def forward(self, x):
		return self.log_likelihood(x,None)



	def sample(self,x, prediction_length):
		'''
		:param x:torch.tensor(batch_size, 1, input_size)
		:param prediction_length:
		:return:
		'''
		future_samples = []
		state=None
		new_samples=x
		for i in range(prediction_length):
			outputs,state=self.rnn(new_samples, state)
			new_samples = self.flow.sample(cond=outputs[:,-1].unsqueeze(1))
			future_samples.append(new_samples)
		x=torch.cat(future_samples, dim=1)
		return x









	def log_likelihood(self,x, mu):
		'''
		:param x: torch.tensor(batch_size,prediction_step,input_size)
		:param mu: torch.tensor(batch_size,1)
		:return:
		'''

		rnn_outputs, states= self.rnn(x)
		log_prob = self.flow.log_prob(x[:,1:], rnn_outputs[:,:-1])

		return -log_prob








import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from packaging import version
import pdb



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def _gelu_python(x):
	"""
	Original Implementation of the GELU activation function in Google BERT repo when initially created. For
	information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
	torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in
	torch.nn.functional Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
	"""
	return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
	"""
	Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
	the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
	"""
	return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


if version.parse(torch.__version__) < version.parse("1.4"):
	gelu = _gelu_python
else:
	gelu = F.gelu


def gelu_fast(x):
	return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def _silu_python(x):
	"""
	See Gaussian Error Linear Units (Hendrycks et al., https://arxiv.org/abs/1606.08415) where the SiLU (Sigmoid Linear
	Unit) was originally introduced and coined, and see Sigmoid-Weighted Linear Units for Neural Network Function
	Approximation in Reinforcement Learning (Elfwing et al., https://arxiv.org/abs/1702.03118) and Swish: a Self-Gated
	Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1) where the SiLU was experimented with
	later.
	"""
	return x * torch.sigmoid(x)


if version.parse(torch.__version__) < version.parse("1.7"):
	silu = _silu_python
else:
	silu = F.silu


def mish(x):
	return x * torch.tanh(torch.nn.functional.softplus(x))


def linear_act(x):
	return x


ACT2FN = {
	"relu": F.relu,
	"silu": silu,
	"swish": silu,
	"gelu": gelu,
	"tanh": torch.tanh,
	"gelu_new": gelu_new,
	"gelu_fast": gelu_fast,
	"mish": mish,
	"linear": linear_act,
	"sigmoid": torch.sigmoid,
}



class MLP(nn.Module):
	'''
	Word specific FCNN implementation from:
	https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_gpt2.py
	'''
	def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
		super().__init__()
		nx = config.n_embd
		self.c_fc = Conv1D(n_state, nx)
		self.c_proj = Conv1D(nx, n_state)
		self.act = ACT2FN[config.activation_function]
		self.dropout = nn.Dropout(config.resid_pdrop)

	def forward(self, x):
		h = self.act(self.c_fc(x))
		h2 = self.c_proj(h)
		return self.dropout(h2)


class Conv1D(nn.Module):
	"""
	1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
	Basically works like a linear layer but the weights are transposed.
	Args:
		nf (:obj:`int`): The number of output features.
		nx (:obj:`int`): The number of input features.
	Note:
		When the model is used for forward propagation,
		the last dimension of the input will be operate  
	"""

	def __init__(self, nf, nx):
		super().__init__()
		self.nf = nf
		w = torch.empty(nx, nf)
		nn.init.normal_(w, std=0.02)
		self.weight = nn.Parameter(w)
		self.bias = nn.Parameter(torch.zeros(nf))

	def forward(self, x):
		size_out = x.size()[:-1] + (self.nf,)
		x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
		x = x.view(*size_out)
		return x

class Attention(nn.Module):
	"""
	Args:
		nx (:obj:`int`): The number of embedding feature, e.g., 128, 256, 512 or so
		n_ctx (:obj:`int`): The context length (not sure)
		config (:obj:T.B.D):
	"""
	def __init__(self, nx, n_ctx, config, scale=False):
		super().__init__()
		
		assert nx % config.n_head == 0
		self.register_buffer(
			"bias", torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx)
		)
		self.register_buffer("masked_bias", torch.tensor(-1e4))
		self.n_head = config.n_head
		self.split_size = nx
		self.scale = scale

		self.c_attn = Conv1D(nx * 3, nx) # Kindly reminder: input_size = [..., nx] and output_size = [..., 3 * nx]
		self.c_proj = Conv1D(nx, nx) # Question: what is the use of this self.c_proj?
		self.attn_dropout = nn.Dropout(config.attn_pdrop)
		self.resid_dropout = nn.Dropout(config.resid_pdrop)

	def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
		w = torch.matmul(q, k)
		if self.scale:
			w = w / (float(v.size(-1)) ** 0.5)
		nd, ns = w.size(-2), w.size(-1)
		mask = self.bias[:, :, ns - nd : ns, :ns]
		w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

		if attention_mask is not None:
			# Apply the attention mask
			w = w + attention_mask

		w = nn.Softmax(dim=-1)(w)
		w = self.attn_dropout(w)
		
		# Mask heads if we want to
		if head_mask is not None:
			w = w * head_mask

		outputs = [torch.matmul(w, v)]
		if output_attentions:
			outputs.append(w)
		return outputs # [value, weights]
	
	def merge_heads(self, x):
		x = x.permute(0, 2, 1, 3).contiguous()
		new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
		return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

	def split_heads(self, x, k=False):
		new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
		
		x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
		if k:
			return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
		else:
			return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
		

	def forward(self, x, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
		x = self.c_attn(x) # x -> q, k, v
		query, key, value = x.split(self.split_size, dim=2) # HanGao: wouldn't it be more general if dim=-1?
		
		query = self.split_heads(query)
		
		key = self.split_heads(key, k=True) # k=True for keys which transposes the last two dims
		value = self.split_heads(value)
		# Concat previous key and value tensors 
		if layer_past is not None:
			past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
			key = torch.cat((past_key, key), dim=-1)
			value = torch.cat((past_value, value), dim=-2)

		if use_cache is True:
			present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
		else:
			present = (None,)
		
		attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
		
		a = attn_outputs[0]
		a = self.merge_heads(a)
		
		a = self.c_proj(a)
		a = self.resid_dropout(a)
		
		outputs = [a, present] + attn_outputs[1:]
		return outputs




class Block(nn.Module):
	def __init__(self, n_ctx, config, scale=False):
		super().__init__()
		nx = config.n_embd
		self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
		self.attn = Attention(nx, n_ctx, config, scale)
		self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
		self.mlp = MLP(4 * nx, config)

	def forward(self, x, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
		# Evaluate attention heads
		output_attn = self.attn.forward(
			self.ln_1(x),
			layer_past=layer_past,
			attention_mask=attention_mask,
			head_mask=head_mask,
			use_cache=use_cache,
			output_attentions=output_attentions,
		)
		a = output_attn[0]  # output_attn: a, present, (attentions)
		# Residual connection 1
		x = x + a
		# FCNN
		m = self.mlp(self.ln_2(x))
		# Residual connection 2
		x = x + m

		outputs = [x] + output_attn[1:]
		return outputs  # x, present, (attentions)


#MLPDense(config.paraEnrichDim, config.n_embd,[200, 200], True)
class MLPDense(torch.nn.Module):
	def __init__(self,nIn,nOut,Hidlayer, withReLU):
		super(MLPDense, self).__init__()
		#print('use Token Model')
		numHidlayer=len(Hidlayer)
		net=[]
		net.append(torch.nn.Linear(nIn,Hidlayer[0]))
		if withReLU:
			net.append(torch.nn.ReLU())
		for i in range(0,numHidlayer-1):
			net.append(torch.nn.Linear(Hidlayer[i],Hidlayer[i+1]))
			if withReLU:
				net.append(torch.nn.ReLU())
		net.append(torch.nn.Linear(Hidlayer[-1],nOut))#
		self.mlp=torch.nn.Sequential(*net)
	def forward(self,x):
		return self.mlp(x)

def enrich_para(mu):
    # parameter
    #ReAll = torch.from_numpy(np.linspace(300, 1000, 101)).float().cuda().reshape([-1,1])
    #pdb.set_trace()
    ReAll = mu
    nuAll = 1/ReAll
    #augment order
    P = 3; dP = 1
    # index of interest
    #I = [i for i in range(101) if i % 2 == 0]
    I = [i for i in range(mu.size(0))]
    # listCatALL
    listCatALL = []
    for i in range(P):
        re = ReAll**(i*dP+1)
        nu = nuAll**(i*dP+1)
        listCatALL.append(re/re.max())
        listCatALL.append(nu/nu.max())
    paraAUG = torch.cat(listCatALL,dim=1)

    # listCatTrain
    paraOI = paraAUG[I,:]

    return paraOI

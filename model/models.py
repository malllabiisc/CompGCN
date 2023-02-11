from helper import *
from model.compgcn_conv import CompGCNConv
from model.compgcn_conv_basis import CompGCNConvBasis

class BaseModel(torch.nn.Module):
	def __init__(self, params):
		super(BaseModel, self).__init__()

		self.p		= params
		self.act	= torch.tanh
		self.bceloss	= torch.nn.BCELoss()

	def loss(self, pred, true_label):
		return self.bceloss(pred, true_label)
		
class CompGCNBase(BaseModel):
	def __init__(self, edge_index, edge_type, num_rel, params=None):
		super(CompGCNBase, self).__init__(params)

		self.edge_index		= edge_index
		self.edge_type		= edge_type
		self.p.gcn_dim		= self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
		self.init_embed		= get_param((self.p.num_ent,   self.p.init_dim))
		self.device		= self.edge_index.device

		if self.p.num_bases > 0:
			self.init_rel  = get_param((self.p.num_bases,   self.p.init_dim))
		else:
			if self.p.score_func == 'transe': 	self.init_rel = get_param((num_rel,   self.p.init_dim))
			else: 					self.init_rel = get_param((num_rel*2, self.p.init_dim))

		if self.p.num_bases > 0:
			self.conv1 = CompGCNConvBasis(self.p.init_dim, self.p.gcn_dim, num_rel, self.p.num_bases, act=self.act, params=self.p)
			self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None
		else:
			self.conv1 = CompGCNConv(self.p.init_dim, self.p.gcn_dim,      num_rel, act=self.act, params=self.p)
			self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

	def forward_base(self, sub, rel, drop1, drop2):

		r	= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
		x, r	= self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r)
		x	= drop1(x)
		x, r	= self.conv2(x, self.edge_index, self.edge_type, rel_embed=r) 	if self.p.gcn_layer == 2 else (x, r)
		x	= drop2(x) 							if self.p.gcn_layer == 2 else x

		sub_emb	= torch.index_select(x, 0, sub)
		rel_emb	= torch.index_select(r, 0, rel)

		return sub_emb, rel_emb, x


class CompGCN_TransE(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.drop, self.drop)
		obj_emb				= sub_emb + rel_emb

		x	= self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)		
		score	= torch.sigmoid(x)

		return score

class CompGCN_DistMult(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.drop, self.drop)
		obj_emb				= sub_emb * rel_emb

		x = torch.mm(obj_emb, all_ent.transpose(1, 0))
		x += self.bias.expand_as(x)

		score = torch.sigmoid(x)
		return score

class CompGCN_ConvE(CompGCNBase):
	def __init__(self, edge_index, edge_type, params=None):
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)

		self.bn0		= torch.nn.BatchNorm2d(1)
		self.bn1		= torch.nn.BatchNorm2d(self.p.num_filt)
		self.bn2		= torch.nn.BatchNorm1d(self.p.embed_dim)
		
		self.hidden_drop	= torch.nn.Dropout(self.p.hid_drop)
		self.hidden_drop2	= torch.nn.Dropout(self.p.hid_drop2)
		self.feature_drop	= torch.nn.Dropout(self.p.feat_drop)
		self.m_conv1		= torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz), stride=1, padding=0, bias=self.p.bias)

		flat_sz_h		= int(2*self.p.k_w) - self.p.ker_sz + 1
		flat_sz_w		= self.p.k_h 	    - self.p.ker_sz + 1
		self.flat_sz		= flat_sz_h*flat_sz_w*self.p.num_filt
		self.fc			= torch.nn.Linear(self.flat_sz, self.p.embed_dim)

	def concat(self, e1_embed, rel_embed):
		e1_embed	= e1_embed. view(-1, 1, self.p.embed_dim)
		rel_embed	= rel_embed.view(-1, 1, self.p.embed_dim)
		stack_inp	= torch.cat([e1_embed, rel_embed], 1)
		stack_inp	= torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2*self.p.k_w, self.p.k_h))
		return stack_inp

	def forward(self, sub, rel):

		sub_emb, rel_emb, all_ent	= self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)
		stk_inp				= self.concat(sub_emb, rel_emb)
		x				= self.bn0(stk_inp)
		x				= self.m_conv1(x)
		x				= self.bn1(x)
		x				= F.relu(x)
		x				= self.feature_drop(x)
		x				= x.view(-1, self.flat_sz)
		x				= self.fc(x)
		x				= self.hidden_drop2(x)
		x				= self.bn2(x)
		x				= F.relu(x)

		x = torch.mm(x, all_ent.transpose(1,0))
		x += self.bias.expand_as(x)

		score = torch.sigmoid(x)
		return score

class CompGCN_ConvKB(CompGCNBase):
	'''Implements the ConvKB Scoring Function'''
	def __init__(self, edge_index, edge_type, params=None):
		'''Init the ComGCN Base and all the  layer used in the Score function ConvKB'''
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)
		self.relu = torch.nn.ReLU()
		self.conv2d1 = torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(1, 3), stride=1, bias=True)
		self.fake_fc_with_conv = torch.nn.Conv2d(self.p.num_filt,out_channels = 1, kernel_size=(self.p.embed_dim, 1), stride=self.p.embed_dim, bias=False)

		#Prints the number of parameter in the network to the console
		total_params = sum(
			param.numel() for param in self.parameters()
		)
		print("Total Parameter: " + str(total_params))

	def forward(self, sub,rel):
		sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.drop, self.drop)
		l = len(sub)
		#we create an array that repeats sub_emb num_ent(Number of Entities) times, in 2 Dimensions
		print(sub_emb.size())
		sub_emb_repeat = sub_emb.unsqueeze(1).expand(-1, self.p.num_ent, -1).reshape((l, 1, self.p.num_ent * self.p.embed_dim))
		# we create an array that repeats rel_emb num_ent(Number of Entities) times, in 2 Dimensions
		rel_emb_repeat = rel_emb.unsqueeze(1).expand(-1, self.p.num_ent, -1).reshape((l, 1, self.p.num_ent * self.p.embed_dim))
		# we reshape all_ent to be in 2 Dimensions
		all_ent = all_ent.unsqueeze(0).expand(l, -1, -1).view((l, 1, self.p.num_ent * self.p.embed_dim))
										#size of tensor after manipulation, bs= batch_size, num_ent= Number of entities, embed_dim = embedding dimension
		#we concatenated the 3 tensors to one tensor
		x = torch.cat([sub_emb_repeat, rel_emb_repeat, all_ent], 1) 	#bs x 3 x num_ent * embed_dim
		x = x.transpose(1, 2) 						#bs x num_ent * embed_dim x 3
		x = x.unsqueeze(1)  						#bs x 1 x num_ent * embed_dim x 3
		#we apply a convolution with a 1x3 kernel
		conv_out = self.conv2d1(x) 					#bs x num_filt x num_ent * embed_dim x 1
		conv_out = self.relu(conv_out)					#bs x num_filt x num_ent * embed_dim x 1

		#we use a convolution with an embed_dim x 1 kernel and a stride of embed_dim and 1 filter, to calculate the dot product of every convolution results of (sub_emb,rel_emb,t) with the same weight w
		score = self.fake_fc_with_conv(conv_out) 			#bs x 1 x num_ent x 1

		score = score.view(l, self.p.num_ent) 				#bs x num_ent
		#we apply a sigmoid function to all dot products
		return torch.sigmoid(score) 					#bs x num_ent

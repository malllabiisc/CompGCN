import torch.nn

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
		self.disable_gnn_encoder = self.p.disable_gnn_encoder

		if self.p.num_bases > 0:
			self.init_rel  = get_param((self.p.num_bases,   self.p.init_dim))
		else:
			if self.p.score_func == 'transe': 	self.init_rel = get_param((num_rel,   self.p.init_dim))
			else: 					self.init_rel = get_param((num_rel*2, self.p.init_dim))

		if not self.disable_gnn_encoder:
			if self.p.num_bases > 0:
				self.conv1 = CompGCNConvBasis(self.p.init_dim, self.p.gcn_dim, num_rel, self.p.num_bases, act=self.act, params=self.p)
				self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None
			else:
				self.conv1 = CompGCNConv(self.p.init_dim, self.p.gcn_dim,      num_rel, act=self.act, params=self.p)
				self.conv2 = CompGCNConv(self.p.gcn_dim,    self.p.embed_dim,    num_rel, act=self.act, params=self.p) if self.p.gcn_layer == 2 else None

		self.register_parameter('bias', Parameter(torch.zeros(self.p.num_ent)))

	def forward_base(self, sub, rel, drop1, drop2):
		"""

		Parameters
		----------
		sub subset
		rel relation
		drop1 first drop function
		drop2 second drop function
		disable_gnn_encoder how to disable the gnn encoder

		Returns subset of the embedding, subset of the relation, embedding
		-------

		"""
		if self.disable_gnn_encoder:
			r = self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
			x = self.init_embed
			x = drop1(x)

			sub_emb = torch.index_select(x, 0, sub)
			rel_emb = torch.index_select(r, 0, rel)
		else:
			r	= self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
			x, r	= self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r)
			x	= drop1(x)
			x, r	= self.conv2(x, self.edge_index, self.edge_type, rel_embed=r) 	if self.p.gcn_layer == 2 else (x, r)
			x	= drop2(x) if self.p.gcn_layer == 2 else x

			sub_emb = torch.index_select(x, 0, sub)
			rel_emb = torch.index_select(r, 0, rel)

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

class CompGCN_CTKGC(CompGCNBase):
		"""
		Implements the CTKGC Scoring Function as in the paper https://link.springer.com/article/10.1007/s10489-021-02438-8 and writes to the console the total parameter used in the network

		The scoring Function works by
			1. Build the Entity-relation matrix by multiplying sub_emb.transpose(1.0) and rel_emb
			2. Then we perform a Convolution on Entity-relation matrix
			3. We Aggregate all the obtained feature,
			4. Convert the above feature maps to a 1D Vector
			5. We project our 1D vector onto a candidate objects to obtain the predicted object embedding
				1. We do 5. for all our entities (candidate objects)
			6. Use the sigmoid function to calculate the score of the object embedding
		"""
		def __init__(self, edge_index, edge_type, params=None):
			"""Init ComGCN Base and all the layers used in the Scoring function CTKGC"""
			super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)

			self.drop = torch.nn.Dropout(self.p.hid_drop)
			self.hid_drop = torch.nn.Dropout(self.p.hid_drop)
			self.feat_drop = torch.nn.Dropout2d(self.p.feat_drop)

			filters = 32
			kernelsize = (3, self.p.embed_dim)
			hidden_size = filters *(self.p.embed_dim - kernelsize[0] +1) * (self.p.embed_dim - kernelsize[1] + 1)
			self.conv2d0 = torch.nn.Conv2d(1, filters, kernelsize, 1, 0, bias=True)

			self.bn0 = torch.nn.BatchNorm2d(1)
			self.bn1 = torch.nn.BatchNorm2d(filters)
			self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)

			self.fc = torch.nn.Linear(hidden_size, self.p.embed_dim)
			total_params = sum(
				param.numel() for param in self.parameters()
			)
			print("Total Parameter: " + str(total_params))


		def forward(self, sub, rel):
			"""

			Parameters
			----------
			sub: idx of entities
			rel: idx of relation

			Returns
			-------
			a list of entities and the percentage of there being a relation of rel between sub and our entities

			Inline comments describe the size of the tensor after manipulation: bs = bach_size
			"""

			sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.drop,self.drop)
			sub_emb = sub_emb.view(-1, self.p.embed_dim, 1)
			rel_emb = rel_emb.view(-1, 1, self.p.embed_dim)

			x = torch.bmm(sub_emb,rel_emb)

			x = x.view(-1, 1, self.p.embed_dim, self.p.embed_dim) 		# bs x 1 x self.p.embed_dim x self.p.embed_dim

			conv_in = self.bn0(x)
			conv_out = self.conv2d0(conv_in) 							# bs x filters x self.p.embed_dim x 1
			conv_out = self.bn1(conv_out)
			conv_out = F.relu(conv_out)
			conv_out = self.feat_drop(conv_out)

			linear_in = conv_out.view(conv_out.shape[0],-1) 			# bs x hidden_size
			linear_out = self.fc(linear_in) 							# bs x self.p.embed_dim
			linear_out = self.hid_drop(linear_out)
			linear_out = self.bn2(linear_out)
			linear_out = F.relu(linear_out)

			prediction = torch.mm(linear_out, all_ent.transpose(1,0)) 	# bs x self.p.num_ent
			prediction += self.bias.expand_as(prediction)
			score = torch.sigmoid(prediction)
			return score

class CompGCN_ConvKB(CompGCNBase):
	'''
	Implements the ConvKB Scoring Function and writes to the console the total Parameter used in the Network

	The scoring Function works by, first concatenating the embedding of the head,relation and tail.
	Then we apply a convolution, and finally we calculate the dot product of the convolution and the weight w.
	'''
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
		'''

		Parameters
		----------
		sub: idx of entities
		rel: idx of relation

		Returns
		-------
		a list of entities and the percentage of there being a relation of rel between sub and our entities

		we first create an array that repeats sub_emb and rel_emb, num_ent(Number of Entities) times
		then we reshape all our array sub_emb_repeat, rel_emb_repeat and all_ent to fit
			1. concatenated the three tensors/arrays
			2. apply a convolution with a 1x3 kernel
			3. apply relu
			4. we calculate the dot product of a weight w (which is same for all entities) and the output of the convolution,
			   we do this by using a convolution of embed_dim x 1 kernel, a stride of embed_dim and one filter
			5. Then we apply a sigmoid function to the output

		Inline comments describe the size of the tensor after manipulation with:
			bs = batch_size
			num_ent = Number of entities
			embed_dim = embedding dimension
		'''
		sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.drop, self.drop)
		l = len(sub)
		sub_emb_repeat = sub_emb.unsqueeze(1).expand(-1, self.p.num_ent, -1).reshape((l, 1, self.p.num_ent * self.p.embed_dim))
		rel_emb_repeat = rel_emb.unsqueeze(1).expand(-1, self.p.num_ent, -1).reshape((l, 1, self.p.num_ent * self.p.embed_dim))
		all_ent = all_ent.unsqueeze(0).expand(l, -1, -1).view((l, 1, self.p.num_ent * self.p.embed_dim))

		x = torch.cat([sub_emb_repeat, rel_emb_repeat, all_ent], 1) 	#bs x 3 x num_ent * embed_dim
		x = x.transpose(1, 2) 											#bs x num_ent * embed_dim x 3
		x = x.unsqueeze(1)  											#bs x 1 x num_ent * embed_dim x 3
		conv_out = self.conv2d1(x) 										#bs x num_filt x num_ent * embed_dim x 1
		conv_out = self.relu(conv_out)									#bs x num_filt x num_ent * embed_dim x 1
		score = self.fake_fc_with_conv(conv_out) 						#bs x 1 x num_ent x 1
		score = score.view(l, self.p.num_ent) 							#bs x num_ent

		return torch.sigmoid(score) 									#bs x num_ent

class CompGCN_Unstructured(CompGCNBase):
	"""
	Implements the Unstructured scoring function.

	The Unstructured scoring function cannot take relationships into account, so it assumes, that head and tail entity
	vectors are similar.
	"""
	def __init__(self, edge_index, edge_type, params=None):
		"""
		Init the CompGCN Base and all the layer used in the score function Unstructured.
		"""
		super(self.__class__, self).__init__(edge_index, edge_type, params.num_rel, params)
		self.drop = torch.nn.Dropout(self.p.hid_drop)

	def forward(self, sub, rel):
		"""

		Parameters
		----------
		sub: idx of entities
		rel: idx of relation

		Returns
		-------
		a list of entities and the percentage of there being a relation of rel between sub and our entities

		We simply subtract the vector of the tail entity from the vector of the head entity.
		"""
		sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.drop, self.drop)

		x = self.p.gamma - torch.norm(sub_emb.unsqueeze(1) - all_ent, p=1, dim=2)
		score = torch.sigmoid(x)
		return score

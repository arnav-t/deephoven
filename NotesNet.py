import torch
from torch import nn

class NotesNet(nn.Module):
	def __init__(self, hidden_dims, device):
		super().__init__()
		self.rnn = nn.GRUCell(input_size=hidden_dims, hidden_size=hidden_dims).to(device)
		self.softmax = nn.LogSoftmax(dim=-1).to(device)

	def forward(self, inp, h):
		h = self.rnn(inp, h)
		h = self.softmax(h)
		return h

	def initHidden(self, hidden_dims):
		hid = torch.randn(1, hidden_dims)
		inp = torch.zeros(1, hidden_dims)
		inp[0] = 1
		return inp, hid

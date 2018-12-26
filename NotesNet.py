import torch
from torch import nn

class NotesNet(nn.Module):
	def __init__(self, hidden_dims, device):
		super().__init__()
		self.rnn = nn.LSTM(input_size=hidden_dims, hidden_size=hidden_dims, batch_first=True).to(device)
		self.softmax = nn.LogSoftmax(dim=-1).to(device)

	def forward(self, x):
		x, _ = self.rnn(x)
		x = self.softmax(x)
		return x

	def initHidden(self, batch_size, hidden_dims):
		ih = torch.zeros(batch_size, 1, hidden_dims)
		ih[0][0][0] = 1
		return ih

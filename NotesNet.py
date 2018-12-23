from torch import nn
from torch import sigmoid

class NotesNet(nn.Module):
	def __init__(self, input_dims, output_dims, device):
		super().__init__()
		self.lstm = nn.LSTM(input_size=input_dims, hidden_size=output_dims, num_layers=2, batch_first=True).to(device)

	def forward(self, x):
		x, _ = self.lstm(x)
		return sigmoid(x)


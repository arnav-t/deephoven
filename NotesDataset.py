from torch import load
from torch.utils.data import Dataset

class NotesDataset(Dataset):
	def __init__(self, fileName, device):
		self.yData = load(fileName).to(device)
		self.len = self.yData.shape[0]

	def __getitem__(self, index):
		return self.yData[index]

	def __len__(self):
		return self.len


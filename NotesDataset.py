import torch
from torch.utils.data import Dataset

def toOneHot(note):
	oneHot = torch.zeros(128)
	oneHot[note] = 1
	return oneHot

def cvtData(data):
	ohData = torch.tensor([])
	for j in range( len(data) ):
		inp = data[j]
		inpTensor = torch.tensor([])
		for i in range( len(inp) ):
			note = int(inp[i])
			oneHot = toOneHot(note)
			inpTensor = torch.cat(( inpTensor, oneHot.unsqueeze(0) ))
		ohData = torch.cat(( ohData, inpTensor.unsqueeze(0) ))
	return ohData

class NotesDataset(Dataset):
	def __init__(self, fileName, device):
		self.yData = torch.load(fileName).to(device) 
		self.len = self.yData.shape[0]

	def __getitem__(self, index):
		return self.yData[index]

	def __len__(self):
		return self.len


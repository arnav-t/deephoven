import numpy as np
import torch
from NotesNet import NotesNet
from NotesDataset import toOneHot
from dataToCSV import writeToFile

BATCH_SIZE = 1
SEQ_LEN = 64
HIDDEN_DIMS = 128
STATE_DICT = 'state.pt'

def toKey(oneHot):
	_, idx = torch.topk(oneHot, 1)
	return int(idx)

def generate(model, device):
	keys = []
	with torch.no_grad():
		inp, output = model.initHidden(HIDDEN_DIMS)
		inp = inp.to(device)
		output = output.to(device)
		for i in range(SEQ_LEN):
			if i >= 1:
				inp = toOneHot(keys[i-1]).unsqueeze(0).to(device)
			output = model(inp, output)
			keys.append( toKey(output[0]) )
	writeToFile('out.csv', keys, True)

if __name__ == '__main__':
	device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
	model = NotesNet(HIDDEN_DIMS, device)
	model.load_state_dict( torch.load(STATE_DICT) )
	generate(model, device)


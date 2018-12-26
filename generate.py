import numpy as np
import torch
from NotesNet import NotesNet
from dataToCSV import writeToFile

BATCH_SIZE = 1
SEQ_LEN = 64
HIDDEN_DIMS = 128
STATE_DICT = 'state.pt'

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

model = NotesNet(HIDDEN_DIMS, device)
model.load_state_dict( torch.load(STATE_DICT) )

def toKey(oneHot):
	_, idx = torch.topk(oneHot, 1)
	return int(idx)

def generate():
	keys = []
	with torch.no_grad():
		output = model.initHidden(BATCH_SIZE, HIDDEN_DIMS).to(device)
		for i in range(SEQ_LEN):
			output = model(output)
			keys.append( toKey(output[0][0]) )
	writeToFile('out.csv', keys, True)

if __name__ == '__main__':
	generate()


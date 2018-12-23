import numpy as np
import torch
from NotesNet import NotesNet
from dataToCSV import writeToFile

BATCH_SIZE = 1
INPUT_DIMS = 64
OUTPUT_DIMS = 64
STATE_DICT = 'state.pt'

KEY_FAC = 100.0

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

model = NotesNet(INPUT_DIMS, OUTPUT_DIMS, device)
model.load_state_dict( torch.load(STATE_DICT) )

def generate():
	input = torch.randn(1, BATCH_SIZE, INPUT_DIMS).to(device)
	model.eval()
	output = model(input)

	output = output.cpu().detach().numpy().squeeze()
	output = np.floor( np.multiply(output, KEY_FAC) ).astype(int)

	writeToFile('out.csv', output, True)



if __name__ == '__main__':
	generate()


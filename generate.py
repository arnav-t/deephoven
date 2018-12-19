import numpy as np
import torch
from NotesNet import NotesNet
from dataToCSV import writeToFile

BATCH_SIZE = 1
NOTES_PER_INPUT = 64
INPUT_DIMS = 1
OUTPUT_DIMS = 3
STATE_DICT = 'state.pt'

TIME_FAC = 10000.0
KEY_FAC = 100.0

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

model = NotesNet(INPUT_DIMS, OUTPUT_DIMS, device)
model.load_state_dict( torch.load(STATE_DICT) )

def generate():
	input = torch.randn(BATCH_SIZE, NOTES_PER_INPUT, INPUT_DIMS).to(device)
	model.eval()
	output = model(input)
	output = output.cpu().detach().numpy().squeeze()
	
	correction = np.amin(output, axis=0)
	correction[1] = 0
	correction[2] = 0

	correction = np.repeat(np.expand_dims(correction, axis=0), NOTES_PER_INPUT, axis=0)
	output = np.subtract(output, correction)

	facMat = np.array([[TIME_FAC, 0, 0], [0, KEY_FAC, 0], [0, 0, KEY_FAC]])
	output = np.floor( np.matmul(output, facMat) )
	output = output[ np.argsort(output[:, 0]) ].astype(int)
	
	writeToFile('out.csv', output, False)



if __name__ == '__main__':
	generate()


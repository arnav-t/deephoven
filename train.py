import torch
import torch.nn as nn
import numpy as np
from NotesDataset import NotesDataset
from NotesNet import NotesNet

DATASET_FILE = './data/data.pt'
BATCH_SIZE = 32
INPUT_DIMS = 64
OUTPUT_DIMS = 64
STATE_DICT = 'state.pt'

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

dataset = NotesDataset(DATASET_FILE, device)
trainLoader = torch.utils.data.DataLoader(
	dataset=dataset,
	batch_size=BATCH_SIZE,
	shuffle=False
)

model = NotesNet(INPUT_DIMS, OUTPUT_DIMS, device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam( model.parameters() )
# inp = np.divide(np.arange(NOTES_PER_INPUT)[np.newaxis].T, NOTES_PER_INPUT).astype(float)
# inp = torch.tensor( np.repeat(np.expand_dims(inp, axis=0), BATCH_SIZE, axis=0) )
# inp = torch.zeros(BATCH_SIZE, NOTES_PER_INPUT, INPUT_DIMS).to(device)


def train(epoch):
	model.train()
	for batchNum, target in enumerate(trainLoader):
		if target.size()[0] != BATCH_SIZE:
			continue
		target = target.to(device)
		inp = torch.randn(1, BATCH_SIZE, INPUT_DIMS).to(device)
		optimizer.zero_grad()
		output = model(inp)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		if batchNum%20 == 0:
			print(f'Epoch: {epoch}, Batch: {batchNum}, Loss: {loss}')

if __name__ == '__main__':
	for epoch in range(1,5):
		train(epoch)
	torch.save( model.state_dict(), STATE_DICT )

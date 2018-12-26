import torch
import torch.nn as nn
import numpy as np
from NotesDataset import NotesDataset
from NotesNet import NotesNet

DATASET_FILE = './data/data.pt'
BATCH_SIZE = 1
SEQ_LEN = 64
HIDDEN_DIMS = 128
STATE_DICT = 'state.pt'

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

print('Loading dataset...')
dataset = NotesDataset(DATASET_FILE, device)
print('Dataset loaded.\n')
trainLoader = torch.utils.data.DataLoader(
	dataset=dataset,
	batch_size=BATCH_SIZE,
	shuffle=False
)

print('Loading model...')
model = NotesNet(HIDDEN_DIMS, device)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam( model.parameters() )
print('Model loaded.\n')

print('Training...')
def train(epoch):
	model.train()
	for batchNum, target in enumerate(trainLoader):
		if target.size()[0] != BATCH_SIZE:
			continue
		target = target.to(device)
		
		loss = 0
		optimizer.zero_grad()
		output = model.initHidden(BATCH_SIZE, HIDDEN_DIMS).to(device)
		for i in range(SEQ_LEN):
			output = model(output)
			loss += criterion(output.squeeze(0), target[0][i].long().unsqueeze(0))
		loss.backward()
		optimizer.step()

		if batchNum%20 == 0:
			print(f'Epoch: {epoch}, Batch: {batchNum}, Loss: {loss}')

if __name__ == '__main__':
	for epoch in range(1,2):
		train(epoch)
	print(f'\nSaving to {STATE_DICT}.')
	torch.save( model.state_dict(), STATE_DICT )

import torch
import torch.nn as nn
import numpy as np
from NotesDataset import NotesDataset
from NotesDataset import toOneHot
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
	shuffle=True
)

print('Loading model...')
model = NotesNet(HIDDEN_DIMS, device)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam( model.parameters() )
print('Model loaded.\n')

print('Training...')
def train(epoch):
	model.train()
	loss = 0
	for batchNum, target in enumerate(trainLoader):
		if target.size()[0] != BATCH_SIZE:
			continue
		target = target.to(device)
		
		optimizer.zero_grad()
		inp, output = model.initHidden(HIDDEN_DIMS)
		inp = inp.to(device)
		output = output.to(device)
		for i in range(SEQ_LEN - 1):
			if i >= 1:
				inp = toOneHot( int(target[0][i]) ).unsqueeze(0).to(device)
			output = model(inp, output)
			loss += criterion(output, target[0][i+1].long().unsqueeze(0))

		if batchNum%20 == 0:
			loss.backward()
			optimizer.step()
			print(f'Epoch: {epoch}, Batch: {batchNum}, Loss: {loss}')
			loss = 0

if __name__ == '__main__':
	for epoch in range(1,2):
		train(epoch)
	print(f'\nSaving to {STATE_DICT}.')
	torch.save( model.state_dict(), STATE_DICT )

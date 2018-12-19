import torch
import torch.nn as nn
from NotesDataset import NotesDataset
from NotesNet import NotesNet

DATASET_FILE = './data/data.pt'
BATCH_SIZE = 64
NOTES_PER_INPUT = 64
INPUT_DIMS = 1
OUTPUT_DIMS = 3
STATE_DICT = 'state.pt'

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

dataset = NotesDataset(DATASET_FILE, device)
trainLoader = torch.utils.data.DataLoader(
	dataset=dataset,
	batch_size=BATCH_SIZE,
	shuffle=True
)

model = NotesNet(INPUT_DIMS, OUTPUT_DIMS, device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam( model.parameters() )

def train(epoch):
	model.train()
	for batchNum, target in enumerate(trainLoader):
		if target.size()[0] != BATCH_SIZE:
			continue
		target = target.to(device)
		input = torch.randn(BATCH_SIZE, NOTES_PER_INPUT, INPUT_DIMS).to(device)
		optimizer.zero_grad()
		output = model(input)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		if batchNum%10 == 0:
			print(f'Epoch: {epoch}, Batch: {batchNum}, Loss: {loss}')

if __name__ == '__main__':
	for epoch in range(1,20):
		train(epoch)
	torch.save( model.state_dict(), STATE_DICT )

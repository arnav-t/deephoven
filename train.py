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
	batch_size=BATCH_SIZE
)

model = NotesNet(INPUT_DIMS, OUTPUT_DIMS)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam( model.parameters() )

def train(epoch):
	model.train()
	for batchNum, target in enumerate(trainLoader):
		target = target.to(device)
		input = torch.randn(BATCH_SIZE, NOTES_PER_INPUT, INPUT_DIMS)
		optimizer.zero_grad()
		output = model(input)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		if batchNum % 20 == 0:
			print(f'Epoch: {epoch}, Batch: {batchNum}, Loss: {loss}')

if __name__ == '__main__':
	for epoch in range(1,5):
		train(epoch)
		torch.save( model.state_dict(), STATE_DICT )

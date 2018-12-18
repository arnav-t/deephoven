import sys
import os
import torch

BATCH_SIZE = 64

whitelist = [
	'Note_on_c'
]

def processCSV(inFile, outFile):
	data = torch.tensor([])
	with open(inFile, 'r') as inCSV:
		batchLen = 0
		offset = 0
		trackNum = '1'
		batch = torch.tensor([])
		for line in inCSV:
			if any(substring in line for substring in whitelist):
				batchLen += 1
				cells = line.strip().split(',')

				if batchLen >= BATCH_SIZE:
					data = torch.cat(( data, batch.unsqueeze(0) ))
					batch = torch.tensor([])
					batchLen = 0
					offset = int(cells[1])
				if cells[0]	!= trackNum:
					trackNum = cells[0]
					batch = torch.tensor([])
					batchLen = 0
					offset = int(cells[1])

				time = float(cells[1]) - offset
				batch = torch.cat(( batch, torch.tensor([[time, float(cells[4]), float(cells[5])]]) ))
	
	outData = data
	if os.path.isfile(outFile):
		outData = torch.cat(( torch.load(outFile), outData ))

	print( f'{len(data)} inputs added. ({BATCH_SIZE*len(data)} notes) [{len(outData)} total inputs]' )

	torch.save(outData, outFile)


if __name__ == '__main__':
	if len(sys.argv) != 2:
		processCSV('out.csv', 'test.pt')
	else:
		processCSV(sys.argv[1], 'data.pt')
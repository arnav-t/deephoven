import sys
import os
import torch

BATCH_SIZE = 64

whitelist = [
	'Note_on_c'
]

KEY_FAC = 100.0

def processCSV(inFile, outFile):
	data = torch.tensor([])
	with open(inFile, 'r') as inCSV:
		batchLen = 0
		trackNum = '1'
		batch = torch.tensor([])
		for line in inCSV:
			if any(substring in line for substring in whitelist):
				cells = line.strip().split(',')
			
				if cells[-1] == ' 0':
					continue

				batchLen += 1

				if batchLen >= BATCH_SIZE:
					data = torch.cat(( data, batch.unsqueeze(0) ))
					batch = torch.tensor([])
					batchLen = 0
				if cells[0]	!= trackNum:
					trackNum = cells[0]
					batch = torch.tensor([])
					batchLen = 0

				batch = torch.cat(( batch, torch.tensor([float(cells[4])/KEY_FAC]) ))
	
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
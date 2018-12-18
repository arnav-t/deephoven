import sys

whitelist = [
	'Start_track',
	'End_track',
	'Note_on_c'
]

def stripCSV(inFile, outFile):
	with open(inFile, 'r') as inCSV, open(outFile, 'w') as outCSV:
		for line in inCSV:
			if any(substring in line for substring in whitelist) or line[0:2] == '0,':
				outCSV.write(line)


if __name__ == '__main__':
	if len(sys.argv) != 2:
		stripCSV('turk.csv', 'test.csv')
	else:
		stripCSV(sys.argv[1], 'out.csv')
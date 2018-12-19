import numpy as np

def writeToFile(fileName, data, verbose=False):
	if verbose:
		print(data)
	with open(fileName, 'w') as file:
		file.write('0, 0, Header, 1, 1, 480\n')
		file.write('1, 0, Start_track\n')
		for note in data:
			file.write(f'1, {note[0]}, Note_on_c, 0, {note[1]}, {note[2]}\n')
		file.write(f'1, {data[-1][0]}, End_track\n')
		file.write('0, 0, End_of_file')

if __name__ == '__main__':
	import random
	testData = []
	for i in range(64):
		testData.append( [i*80, random.randint(60,80), random.randint(60,80)] )
	testData = np.array(testData)
	writeToFile('out.csv', testData, False)
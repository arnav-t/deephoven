import numpy as np

NOTES_PER_INPUT = 64
NOTE_DELAY = 200

def writeToFile(fileName, data, verbose=False):
	if verbose:
		print(data)
	with open(fileName, 'w') as file:
		file.write('0, 0, Header, 1, 1, 480\n')
		file.write('1, 0, Start_track\n')
		for step, note in enumerate(data):
			file.write(f'1, {step*NOTE_DELAY}, Note_on_c, 0, {note}, 50\n')
			file.write(f'1, {(step+1)*NOTE_DELAY}, Note_on_c, 0, {note}, 0\n')
		file.write(f'1, {NOTE_DELAY*64}, End_track\n')
		file.write('0, 0, End_of_file')

if __name__ == '__main__':
	import random
	testData = []
	for i in range(NOTES_PER_INPUT):
		testData.append( random.randint(60,80) )
	testData = np.array(testData)
	writeToFile('out.csv', testData, False)
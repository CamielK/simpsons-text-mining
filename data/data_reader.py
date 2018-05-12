import csv
import os

def get_script_lines(startOffset = 0, numRecords = 0):
    return load_csv('simpsons_script_lines.csv', startOffset, numRecords)
def get_characters(startOffset = 0, numRecords = 0):
    return load_csv('simpsons_characters.csv', startOffset, numRecords)
def get_episodes(startOffset = 0, numRecords = 0):
    return load_csv('simpsons_episodes.csv', startOffset, numRecords)
def get_locations(startOffset = 0, numRecords = 0):
    return load_csv('simpsons_locations.csv', startOffset, numRecords)

def load_csv(filename, startOffset, numRecords):
    data = []
    filerealname = os.path.abspath(os.path.realpath(__file__) + '/../' + filename)
    with open(filerealname) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in readCSV:
            i += 1
            if (startOffset == 0 and i == 1): continue  # Skip csv column header
            if (i >= startOffset and i < startOffset + numRecords): data.append(row)
            if (numRecords > 0 and i > numRecords): break
    return data

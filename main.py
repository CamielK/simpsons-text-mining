import spacy as sp
from spacy import displacy
from model.script_line import ScriptLine
import csv

nlp = sp.load('en')

if __name__ == "__main__":

    # Load first 100 records
    data = []
    with open('data/simpsons_script_lines.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        i = 0
        for row in readCSV:
            i += 1
            if (i == 1): continue  # Skip csv column header
            data.append(row)
            if (i > 100): break

    # Basic processing of script lines
    for record in data:
        script_line = ScriptLine(record)
        doc = nlp(script_line.spoken_words)
        print('\n\nProcessing: ', script_line.spoken_words)
        print('%-12s%-12s%-12s%-12s%-12s%-12s%-12s%-12s' % ('text','lemma','POS','tag','dep','shape','is_alpha','is_stop'))
        for token in doc:
            print('%-12s%-12s%-12s%-12s%-12s%-12s%-12s%-12s' % (token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop))
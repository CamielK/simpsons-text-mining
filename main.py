import spacy as sp
from model.script_line import ScriptLine
from data import data_reader as dr

nlp = sp.load('en')

if __name__ == "__main__":

    # Load first 100 script lines
    data = dr.get_script_lines(0, 100)

    # Basic processing of script lines
    for record in data:
        script_line = ScriptLine(record)
        doc = nlp(script_line.spoken_words)
        print('\n\nProcessing: ', script_line.spoken_words)
        print('%-12s%-12s%-12s%-12s%-12s%-12s%-12s%-12s' % ('text','lemma','POS','tag','dep','shape','is_alpha','is_stop'))
        for token in doc:
            print('%-12s%-12s%-12s%-12s%-12s%-12s%-12s%-12s' % (token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop))
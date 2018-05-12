import pickle
from model.script_line import ScriptLine
from data import data_reader as dr
from sentiment import helper as hlp
from nltk.corpus import stopwords

# Load classifier
f = open('model/sentiment_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()

def get_words_in_script_lines(data):
	all = []
	for script_line in data:
		all.extend(script_line)
	return all

print('applying classifier to test data')
data = dr.get_script_lines(0, 10000)

# Filter corpus
script_lines_raw = []
# stopwords_set = set(stopwords.words("english"))
for record in data:
	script_line = ScriptLine(record)
	words_filtered = [e.lower() for e in script_line.spoken_words.split() if len(e) >= 3]
	# words_without_stopwords = [word for word in words_filtered if not word in stopwords_set]
	script_lines_raw.append(words_filtered)
	# script_lines_raw.append(words_without_stopwords)

w_features = hlp.get_word_features(get_words_in_script_lines(script_lines_raw)) # create a list of all word features in the corpus
for record in data:
	script_line = ScriptLine(record)
	if (len(script_line.spoken_words.split()) > 5): # Only apply classifier if sentence has more then 5 tokens
		res = classifier.classify(hlp.extract_features(script_line.spoken_words.split(), w_features))
		print('%-12s%-12s' % (res, script_line.spoken_words))

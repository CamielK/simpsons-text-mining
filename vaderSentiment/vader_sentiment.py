from model.script_line import ScriptLine
from data import data_reader as dr

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

def get_words_in_script_lines(data):
	all = []
	for script_line in data:
		all.extend(script_line)
	return all

data = dr.get_script_lines(0, 100)

# Filter corpus
script_lines_raw = []
# stopwords_set = set(stopwords.words("english"))
for record in data:
	script_line = ScriptLine(record)
	words_filtered = [e.lower() for e in script_line.spoken_words.split() if len(e) >= 3]
	# words_without_stopwords = [word for word in words_filtered if not word in stopwords_set]
	script_lines_raw.append(words_filtered)
	# script_lines_raw.append(words_without_stopwords)

sid = SentimentIntensityAnalyzer()
for record in data:
	script_line = ScriptLine(record)
	if (len(script_line.spoken_words.split()) > 0): # Only apply classifier if sentence has more than 5 tokens
		ss = sid.polarity_scores(script_line.spoken_words)
		print(script_line.spoken_words + ' - ', end='')
		for k in sorted(ss):
 			print('{0}: {1}, '.format(k, ss[k]), end='')
		print()

import csv

from model.script_line import ScriptLine
from data import data_reader as dr

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

data = dr.get_script_lines()
# data = dr.get_script_lines(0, 10000)
# data = dr.get_script_lines(0, 100)

sid = SentimentIntensityAnalyzer()
i = 0
sentiment = []
for record in data:
	i += 1
	script_line = ScriptLine(record)
	if i % 100 == 0:
		print('Progress: ' + str(i) + '/' + str(len(data)))

	# if (len(script_line.spoken_words.split()) > 5):  # Only apply classifier if sentence has more than 5 tokens

	ss = sid.polarity_scores(script_line.spoken_words)
	# ss = sid.polarity_scores(script_line.spoken_words)

	# for k in sorted(ss):
	# 	print('{0}: {1}, '.format(k, ss[k]), end='')
	# print(script_line.spoken_words)

	if 'compound' in ss:
		# Write sentiment to file
		compound = ss.get('compound')
		negative = ss.get('neg')
		neutral = ss.get('neu')
		positive = ss.get('pos')
		sentiment.append({
			'script_line_id': script_line.id,
			'compound': compound,
			'negative': negative,
			'neutral': neutral,
			'positive': positive,
		})

	else:
		print('Unable to parse sentiment for script line: ' + script_line.id)


#Write to file
print('writing to file')
with open("results_sentiment.csv", "w", newline='') as outfile:
	## Ordering of the fields in the CSV output
	headers = ['script_line_id', 'compound', 'negative', 'neutral', 'positive']

	writer = csv.writer(outfile, delimiter=",", quotechar='', quoting=csv.QUOTE_NONE)
	writer.writerow(headers)

	## Write out the data
	for obj in sentiment:
		writer.writerow([obj[key] for key in headers])
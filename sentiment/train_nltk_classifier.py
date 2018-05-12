import pandas as pd
from sklearn.model_selection import train_test_split
import sentiment.helper as hlp

import nltk
from nltk.corpus import stopwords
import pickle

# Load sentiment training data
data = pd.read_csv('model/sentiment.csv')
data = data[['text', 'sentiment']]

# split test set and drop neutral sentiments
train, test = train_test_split(data, test_size=0.1)
train = train[train.sentiment != "Neutral"]

print('loading test tweets')
tweets = []
stopwords_set = set(stopwords.words("english"))
for index, row in train.iterrows():
	words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
	words_cleaned = [word for word in words_filtered
					 if 'http' not in word
					 and not word.startswith('@')
					 and not word.startswith('#')
					 and word != 'RT']
	words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
	tweets.append((words_cleaned, row.sentiment))

test_pos = test[test['sentiment'] == 'Positive']
test_pos = test_pos['text']
test_neg = test[test['sentiment'] == 'Negative']
test_neg = test_neg['text']

def get_words_in_tweets(doc):
	all = []
	for (words, sentiment) in doc:
		all.extend(words)
	return all

# Training the Naive Bayes classifier
print('training classifier')
training_set = nltk.classify.apply_features(hlp.extract_features,tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)

# Save classifier
f = open('model/sentiment_classifier.pickle', 'wb')
pickle.dump(classifier, f)
f.close()

print('applying classifier to test data')
neg_cnt = 0
pos_cnt = 0
w_features = hlp.get_word_features(get_words_in_tweets(tweets))
for obj in test_neg:
	res = classifier.classify(hlp.extract_features(obj.split(), w_features))
	if (res == 'Negative'):
		neg_cnt = neg_cnt + 1
for obj in test_pos:
	res = classifier.classify(hlp.extract_features(obj.split(), w_features))
	if (res == 'Positive'):
		pos_cnt = pos_cnt + 1

print('[Negative]: %s/%s ' % (len(test_neg), neg_cnt))
print('[Positive]: %s/%s ' % (len(test_pos), pos_cnt))

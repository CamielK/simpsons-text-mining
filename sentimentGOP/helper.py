import nltk


def get_word_features(wordlist):
	wordlist = nltk.FreqDist(wordlist)
	features = wordlist.keys()
	return features


def extract_features(document, w_features):
	document_words = set(document)
	features = {}
	for word in w_features:
		features['containts(%s)' % word] = (word in document_words)
	return features

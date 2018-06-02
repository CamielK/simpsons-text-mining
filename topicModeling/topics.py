class ScriptLine(object):
	def __init__(self, csv_line):
		self.id = csv_line[0]
		self.episode_id = csv_line[1]
		self.number = csv_line[2]
		self.raw_text = csv_line[3]
		self.timestamp_in_ms = csv_line[4]
		self.speaking_line = csv_line[5]
		self.character_id = csv_line[6]
		self.location_id = csv_line[7]
		self.raw_character_text = csv_line[8]
		self.raw_location_text = csv_line[9]
		self.spoken_words = csv_line[10]
		self.normalized_text = csv_line[11]
		self.word_count = csv_line[12]

	tokens = []
	def setPreprocessed(self, cleaned_tokens):
		self.tokens = cleaned_tokens


class Episode(object):
    def __init__(self, csv_line):
        self.id                     = csv_line[0]
        self.title                  = csv_line[1]
        self.original_air_date      = csv_line[2]
        self.production_code        = csv_line[3]
        self.season                 = csv_line[4]
        self.number_in_season       = csv_line[5]
        self.number_in_series       = csv_line[6]
        self.us_viewers_in_millions = csv_line[7]
        self.views                  = csv_line[8]
        self.imdb_rating            = csv_line[9]
        self.imdb_votes             = csv_line[10]
        self.image_url              = csv_line[11]
        self.video_url              = csv_line[12]


import csv
import sys
import os
import gensim
from gensim import corpora, models, similarities
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import operator

nltk.download('punkt')

stop = stopwords.words('english') + list(string.punctuation)

# Read script lines
print("Reading and preprocessing script lines")
script_lines = []
script_lines_raw = []
with open("../data/simpsons_script_lines.csv") as f:
	reader = csv.reader(f)
	next(reader)  # skip header
	for r in reader:
		script_line = ScriptLine(r)
		if script_line.spoken_words != "":
			cleaned_tokens = [i for i in word_tokenize(script_line.spoken_words.lower().decode('utf-8')) if i not in stop and len(i) > 3]
			script_line.setPreprocessed(cleaned_tokens)
			script_lines_raw.append(cleaned_tokens)
		script_lines.append(script_line)

# Read episodes
print("Reading episodes")
characters = []
# with open("characters_main.csv") as f: # only main cast
with open("../data/simpsons_episodes.csv") as f: # all characters
	reader = csv.reader(f)
	next(reader)  # skip header
	for r in reader:
		characters.append(Episode(r))

# Create dictionary
print("Creating dictionary")
dictionary = corpora.Dictionary(script_lines_raw)
# dictionary.save('/simpsons.dict')
print(dictionary.token2id)

# Create corpus
print("Creating corpus")
corpus = [dictionary.doc2bow(line) for line in script_lines_raw]

# Compute TF-IDF
print("Creating TF-IDF model")
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

num_topics = 10
# Build LSA model
print("LSA Model:")
lsa_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
for topic in range(num_topics):
	words = lsa_model.show_topic(topic)
	words_str = ''
	for word in words:
		words_str += ' ' + word[0]
	print('topic ' + str(topic) + ': ' + words_str)

# Build LDA model
print("LDA Model:")
lda_model = models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=num_topics, update_every=1, chunksize=10000, passes=1)
for topic in range(num_topics):
	words = lda_model.get_topic_terms(topic)
	words_str = ''
	for word in words:
		words_str += ' ' + dictionary[word[0]]
	print('topic ' + str(topic) + ': ' + words_str)

print()

def getScriptLinesForCharacter(character_id):
	char_lines = []
	for script_line in script_lines:
		if script_line.character_id == character_id:
			char_lines.append(script_line)
	return char_lines


print("Parsing characters")
character_vocab = {}
for character in characters:
	print("Parsing character: " + character.name)

	# Get all lines for this character
	char_lines = getScriptLinesForCharacter(character.id)

	# Skip characters with minimal lines
	if len(char_lines) < 100:
		continue

	# Calculate TF-IDF for character
	char_lines_raw = ''
	for line in char_lines:
		if len(line.tokens) > 0:
			char_lines_raw += ' ' + (' '.join(line.tokens))

	# Compute character dictionary
	dictionary_character = dictionary.doc2bow(char_lines_raw.lower().split())

	# Calculate tfidf score for each token used by character
	tokens = {}
	tfs = {}
	for token in dictionary_character:
		# TF
		tf = token[1]
		# IDF
		idf = tfidf.idfs.get(token[0])
		# TFIDF
		tfidf_score = tf * idf

		tokens[token[0]] = tfidf_score
		tfs[token[0]] = tf

	# Sort by highest tfidf score
	sorted_tokens = sorted(tokens.items(), key=operator.itemgetter(1), reverse=True)
	sorted_tfs = sorted(tfs.items(), key=operator.itemgetter(1), reverse=True)

	# Get the best tokens and their original tokens
	character_words = []
	i = 0
	for token in sorted_tokens:
		if i < 500:
			# Get top 30 words
			character_words.append(dictionary.get(token[0]))
		i += 1

	# Get the most used tokens
	character_most_used = []
	i = 0
	for token in sorted_tfs:
		if i < 500:
			# Get top 30 words
			character_most_used.append(dictionary.get(token[0]))
		i += 1

	# Compute word2vec vector for character (using only the most used words)
	character_word2vec = PhraseVector(wordvec_model, " ".join(character_words))

	character_vocab[character.id] = {
		'id': character.id,
		'name': character.name,
		'tfidf': character_words,
		'tf': character_most_used,
		'w2v': character_word2vec
	}


# Print results
print("Showing results")
similarity_char_matrix = []
labels = []
for id in character_vocab:
	character = character_vocab[id]
	print()
	print(character['name'])
	print("-".join(character['tfidf']))
	print("-".join(character['tf']))
	print("Most similar to: ")
	labels.append(character['name'])

	# Find most similar characters based on their word2vec vectors
	similarities_char = []
	char_w2v = character['w2v']
	for other_id in character_vocab:
		other_character = character_vocab[other_id]
		other_char_w2v = other_character['w2v']
		similarity_to_char = char_w2v.CosineSimilarity(other_char_w2v.vector)
		similarities_char.append(similarity_to_char)
	similarity_char_matrix.append(similarities_char)

import matplotlib.pyplot as plt
from matplotlib import cm as cm

cmap = cm.get_cmap('YlGnBu')

fig, ax = plt.subplots(figsize=(20, 20))
cax = ax.matshow(similarity_char_matrix, interpolation='nearest', cmap=cmap)
ax.grid(True)
# plt.title('Simpsons Character Similarity matrix')
plt.xticks(range(len(labels)), labels, rotation=90)
plt.yticks(range(len(labels)), labels)
fig.colorbar(cax, ticks=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, .75, .8, .85, .90, .95, 1])
plt.show()

print()
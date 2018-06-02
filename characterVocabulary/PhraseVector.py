import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import numpy as np
import math

class PhraseVector:
	def __init__(self, wordvec_model, phrase):
		self.phrase = phrase
		self.wordvec_model = wordvec_model
		self.vector = self.PhraseToVec(phrase)

	# Retireve original phrase text
	def GetPhrase(self):
		return self.phrase

	# Combine multiple vectors
	def ConvertVectorSetToVecAverageBased(self, vectorSet, ignore=[]):
		if len(ignore) == 0:
			return np.mean(vectorSet, axis=0)
		else:
			return np.dot(np.transpose(vectorSet), ignore)/sum(ignore)

	# Retrieve the phrase vector based on the vectors of each word in the phrase
	def PhraseToVec(self, phrase):
		phrase = phrase.lower()
		wordsInPhrase = [word for word in phrase.split()]
		vectorSet = []
		for aWord in wordsInPhrase:
			try:
				wordVector = self.wordvec_model[aWord]
				vectorSet.append(wordVector)
			except:
				# Word not in vocabulary
				pass
		return self.ConvertVectorSetToVecAverageBased(vectorSet)

	# Calculate the cosine similarity for the current phrase and another phrase vector
	def CosineSimilarity(self, otherPhraseVec):
		cosine_similarity = np.dot(self.vector, otherPhraseVec) / (np.linalg.norm(self.vector) * np.linalg.norm(otherPhraseVec))
		try:
			if math.isnan(cosine_similarity):
				cosine_similarity = 0
		except:
			cosine_similarity = 0
		return cosine_similarity
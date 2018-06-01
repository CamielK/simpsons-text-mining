# The Simpsons text mining


## Sentiment
This python project is used to tag the script lines with their sentiment values using different classifiers
For further processing of this data see the enclosed Java project.

### sentimentGOP
Sentiment model based on GOP debate tweets. NLTK naive bayes classifier is used to classify new text.
Performance is not very good, script lines are mostly tagged as negative (~90%)

### vaderSentiment
vaderSentiment.py uses the nltk vader package to do sentiment analysis. The original data is tagged with these sentiment values.

## Data
Source: https://kaggle.com/wcukierski/the-simpsons-by-the-data (Author: Todd Schneider)

## Requirements
- Python 3.x
- nltk (download models: en, vader)
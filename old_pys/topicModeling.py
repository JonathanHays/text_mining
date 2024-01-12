# -*- coding: utf-8 -*-
"""
Created on Tue Jan 9 18:00:42 2024

@author: jonha
"""

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim import corpora, models
import pyLDAvis.gensim
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Sample data
excel_file_path = 'C:\\TestCode\\csat\\excel\\sampleMachineData.xlsx'

# Load data from Excel file
df = pd.read_excel(excel_file_path)

# Tokenizing and removing stopwords
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return words

df['Tokenized'] = df['Response'].apply(preprocess_text)

# Creating a dictionary and corpus
dictionary = corpora.Dictionary(df['Tokenized'])
corpus = [dictionary.doc2bow(tokens) for tokens in df['Tokenized']]

# LDA model
lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)

# Visualizing the topics
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(vis)

# Plotting Word Clouds for each topic
topics = lda_model.show_topics(num_topics=3, num_words=10, formatted=False)
for topic_id, words in topics:
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f"Topic {topic_id + 1}")
    plt.axis('off')
    plt.show()

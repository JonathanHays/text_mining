import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim
from gensim import corpora
from pprint import pprint

# Assuming you have a 'Responses' column in your DataFrame
# Replace 'your_dataset.csv' with the actual file path or data loading method you are using
excel_file_path = 'C:\\TestCode\\csat\\excel\\sampleMachineData.xlsx'
df = pd.read_excel(excel_file_path)

# Tokenizing and removing stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return words

df['Tokenized'] = df['Response'].apply(preprocess_text)

# Create a dictionary and corpus for LDA
dictionary = corpora.Dictionary(df['Tokenized'])
corpus = [dictionary.doc2bow(tokens) for tokens in df['Tokenized']]

# Train LDA model
lda_model = gensim.models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

# Print topics and associated keywords
pprint(lda_model.print_topics())

# Alternatively, you can visualize the topics using pyLDAvis
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
pyLDAvis.display(vis)

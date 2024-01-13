# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 09:54:43 2024

@author: jonha

Text Analysis Function Only
"""
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import ngrams, sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from afinn import Afinn
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
from collections import Counter
import string
from transformers import pipeline
from tqdm import tqdm
###common variables

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
afinn = Afinn()
sentiment_pipeline = pipeline("sentiment-analysis")

def excel_output(df, output_excel_path):
    # Check if the file exists and remove it
    if os.path.exists(output_excel_path):
        os.remove(output_excel_path)
        
    df.to_excel(output_excel_path, index=False)

    print(f"{df.name} exported to {output_excel_path}")


###Function for single wocount and word sentiment

def preprocess_single_word(text):
    # Check if text is a string or convert to string
    if not isinstance(text, str) or pd.isna(text):
        return []  # Return an empty list for NaN values
    
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stopwords and filter out punctuation and single quotes
    words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
    
    # Calculate sentiment score for each word
    word_sentiments = [afinn.score(word) for word in words]
    
    return list(zip(words, word_sentiments))

def single_word_count(df, column, output_excel_path):
    print("Running Single Word Text Analysis")
    df['WordAndSentiment'] = df[column].apply(preprocess_single_word)
    # Filter out rows where 'WordAndSentiment' is empty
    df_exploded = df.explode('WordAndSentiment').dropna(subset=['WordAndSentiment'])
    # Counting word occurrences
    word_counts = df_exploded['WordAndSentiment'].value_counts().reset_index()
    word_counts.columns = ['WordAndSentiment', 'Count']
    # Split the 'WordAndSentiment' column into separate 'Word' and 'Sentiment' columns
    word_counts[['Word', 'Sentiment']] = pd.DataFrame(word_counts['WordAndSentiment'].tolist(), index=word_counts.index)
    word_counts.name = 'Single Word Counts and Sentiment'
    excel_output(word_counts, output_excel_path)
    
    return word_counts

### Functions for biggrams

def preprocess_biggrams_text(text):
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    return text 

def generate_trigrams(text, size):
    words = word_tokenize(text)
    trigrams = ngrams(words, size)
    return trigrams

def calculate_biggram_sentiment_score(text):
    sentences = sent_tokenize(text)
    analyzer = SentimentIntensityAnalyzer()
    compound_score = 0

    for sentence in sentences:
        compound_score += analyzer.polarity_scores(sentence)['compound']

    return compound_score

def biggrams(df, column, size, output_excel_path):
    print("Running Biggram Text Analysis")
    all_responses = ' '.join(df[column].astype(str))
    all_responses = preprocess_biggrams_text(all_responses)
    trigram_counts = Counter(generate_trigrams(all_responses, size))
    sentiment_scores = {trigram: calculate_biggram_sentiment_score(' '.join(trigram)) for trigram in trigram_counts}
    result_df = pd.DataFrame(list(trigram_counts.items()), columns=['Trigram', 'Count'])
    result_df['SentimentScore'] = result_df['Trigram'].apply(lambda x: sentiment_scores[x])
    result_df['CombinedWords'] = result_df['Trigram'].apply(lambda x: ' '.join(x))
    result_df.name = 'Biggrams Count and Sentiment'
    excel_output(result_df, output_excel_path)
    
    return result_df


###Sentiment pipeline whole response rating



def analyze_response_sentiment(text):
    result = sentiment_pipeline(text)
    return result[0]['label']    
    
    
def process_chat_data(df, output_excel_path):
    
    print("Running Response Text Analysis")
    tqdm.pandas(desc="Applying Sentiment Analysis")
    df['sentiment_score'] = df['chat_survey_response'].progress_apply(analyze_response_sentiment)
    #results_df = df[['chat_transcript_id', 'sentiment_score']].copy()
    df.name = 'Response Sentiment Rating'
    excel_output(df, output_excel_path)
    
    return df


#### Wordcloud functions

def all_words_list(df, sentiment):
     if sentiment == "positive":
         df = df[df['Sentiment'] > 0]
         results = ' '.join(df['Word'].astype(str))       
         return  results
     elif sentiment == "negative":
        df = df[df['Sentiment'] < 0]
        results = ' '.join(df['Word'].astype(str))       
        return  results
     else:
        results = ' '.join(df['Word'].astype(str))       
        return  results

def basic_word_cloud(df, sentiment, output_path, background_color, color_map, maxWords):
    print("Creating Word Cloud")
    word_list = all_words_list(df, sentiment)
    if color_map == "":
        wordcloud = WordCloud(width=1600, height=800, background_color=background_color, max_words= maxWords).generate(word_list)
    else:
        wordcloud = WordCloud(width=1600, height=800, background_color=background_color, max_words= maxWords, colormap = color_map).generate(word_list)
    plt.figure(figsize=(30,20), facecolor='k')
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    wordcloud.to_file(output_path)
    print(f"Wordcloud exported to {output_path}")
 
def custom_word_cloud_image(df, sentiment, output_path, background_color, color_map, maxWords, image):
    print("Creating Word Cloud")
    word_list = all_words_list(df, sentiment)
    mask = np.array(Image.open(image))
    image_colors = ImageColorGenerator(mask)
    if color_map == "":
        wordcloud = WordCloud(mask=mask, width=1600, height=800, background_color=background_color, max_words= maxWords).generate(word_list)
    else:
        wordcloud = WordCloud(mask=mask, width=1600, height=800, background_color=background_color, max_words= maxWords, colormap = color_map).generate(word_list)
    plt.figure(figsize=(30,20), facecolor='k')
    #plt.imshow(wordcloud, interpolation="bilinear")
    plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation = "bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    wordcloud.to_file(output_path)
    print(f"Wordcloud exported to {output_path}")
    
    
def custom_word_cloud_image_color(df, sentiment, output_path, background_color, maxWords, image):
    print("Creating Word Cloud")
    word_list = all_words_list(df, sentiment)
    mask = np.array(Image.open(image))
    image_colors = ImageColorGenerator(mask)
    wordcloud = WordCloud(mask=mask, width=1600, height=800, background_color=background_color, max_words= maxWords).generate(word_list)
    plt.figure(figsize=(30,20), facecolor='k')
    plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation = "bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    wordcloud.to_file(output_path)
    print(f"Wordcloud exported to {output_path}")
    
        
    
    
    
    
    
    
    
    
    
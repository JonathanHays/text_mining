# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 08:50:19 2024

@author: jonha
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_sentiment_score(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)['compound']
    return sentiment_score

def process_chat_data(input_path, output_path):

    chat_data = pd.read_excel(input_path)

    chat_data = chat_data.dropna(subset=['chat_survey_response'])

    chat_data['sentiment_score'] = chat_data['chat_survey_response'].apply(get_sentiment_score)

    results_df = chat_data[['chat_transcript_id', 'sentiment_score']].copy()

    results_df.to_excel(output_path, index=False)

# File paths
input_path = 'C:\\TestCode\\csat\\excel\\sampleCustomerChatData.xlsx'
output_path = 'C:\\TestCode\\csat\\excel\\response_sentiment.xlsx'

# Process chat data and output results
process_chat_data(input_path, output_path)

print(f"Sentiment scores exported to {output_path}")
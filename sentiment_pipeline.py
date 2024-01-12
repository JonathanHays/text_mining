# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 14:20:52 2024

@author: jonha
"""
from transformers import pipeline
import pandas as pd

sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment_pipeline(text)
    return result[0]['label']



def process_chat_data(input_path, output_path):

    chat_data = pd.read_excel(input_path)

    chat_data = chat_data.dropna(subset=['chat_survey_response'])
    print("1")
    chat_data['sentiment_score'] = chat_data['chat_survey_response'].apply(analyze_sentiment)
    print("1")
    results_df = chat_data[['chat_transcript_id', 'sentiment_score']].copy()
    print("1")
    results_df.to_excel(output_path, index=False)
    print("1")
# File paths
input_path = 'C:\\TestCode\\csat\\excel\\sampleCustomerChatData.xlsx'
output_path = 'C:\\TestCode\\csat\\excel\\higgingAnalysis_sentiment.xlsx'

# Process chat data and output results
process_chat_data(input_path, output_path)

print(f"Sentiment scores exported to {output_path}")
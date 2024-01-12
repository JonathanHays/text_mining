# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 09:54:09 2024
@author: jonha
Main File 
"""
import helpers as h
import pandas as pd


## directories
input_dir = 'C:\\TestCode\\csat\\sampleData\\'
input_image_dir = 'C:\\TestCode\\csat\\images\\'
output_dir = 'C:\\TestCode\\csat\\outputData\\'

# inputs
input_excel = 'sampleCustomerChatData.xlsx'
response_column_name = 'chat_survey_response'

# outputs
output_word_count_sentiment = 'word_counts_sentiment.xlsx'
output_biggram_count_sentiment = 'biggram_counts_sentiment.xlsx'
output_response_sentiment = 'response_sentiment.xlsx'
output_wordcloud = 'wordCloud.png'

# Dataframe
chat_data = pd.read_excel(input_dir + input_excel)
chat_data = chat_data.dropna(subset=['chat_survey_response'])


# Uncomment the functions you want to run below. Comment out ones you do not want to run. 
#####

# run this to get a count of all words and their sentiment score

word_counts = h.single_word_count(chat_data, response_column_name, output_dir + output_word_count_sentiment )

# run this for biggram text analysis set the number to the return the desired length of the biggram

#biggram_counts = h.biggrams(chat_data, response_column_name, 3, output_dir + output_biggram_count_sentiment)

# run this for a complete response analysis to determine if the response is positive or negative This one takes a LONG LONG time

#response_rating = h.process_chat_data(chat_data, output_dir + output_response_sentiment)


# The following functions will create wordclouds
# https://matplotlib.org/stable/users/explain/colors/colormaps.html - To view colormap options
# Must run word_counts funtion above for these to run

# Wordcloud variables adjust as needed

max_words = 1000
background_color = "white" # #363838 - dark grey background
color_map = ""  # will use default if left blank
sentiment = ""  # leave blank for all words, negative for negative words, positive for positive words
output = output_dir + output_wordcloud
mask_image = input_image_dir + 'cash_app.jpg'


# Simple wordcloud

#h.basic_word_cloud(word_counts, sentiment , output , background_color , color_map, max_words )

# Custom Shape wordcloud
#h.custom_word_cloud_image(word_counts, sentiment , output , background_color , color_map, max_words, mask_image )

# Custom Shape wordcloud and color based of image color(s)
h.custom_word_cloud_image_color(word_counts, sentiment , output , background_color , max_words, mask_image )
























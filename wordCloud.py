import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from afinn import Afinn
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

excel_file_path = 'C:\\TestCode\\csat\\sampleCustomerChatData.xlsx'

# Creating a DataFrame
df = pd.read_excel(excel_file_path)

# Tokenizing, removing stopwords, and filtering out punctuation and single quotes
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# Create an Afinn object
afinn = Afinn()

def preprocess_text(text):
    # Check if text is a string or convert to string
    if not isinstance(text, str) or pd.isna(text):
        return []  # Return an empty list for NaN values
    
    # Tokenize the text
    words = word_tokenize(text)
    
    # Remove stopwords and filter out punctuation and single quotes
    words = [word for word in words if word.lower() not in stop_words and word.isalpha() and word not in string.punctuation and word != "'"]
    
    # Calculate sentiment score for each word
    word_sentiments = [afinn.score(word) for word in words]
    
    return list(zip(words, word_sentiments))

df['WordAndSentiment'] = df['chat_survey_response'].apply(preprocess_text)

# Filter out rows where 'WordAndSentiment' is empty
df_exploded = df.explode('WordAndSentiment').dropna(subset=['WordAndSentiment'])

# Counting word occurrences
word_counts = df_exploded['WordAndSentiment'].value_counts().reset_index()
word_counts.columns = ['WordAndSentiment', 'Count']

# Split the 'WordAndSentiment' column into separate 'Word' and 'Sentiment' columns
word_counts[['Word', 'Sentiment']] = pd.DataFrame(word_counts['WordAndSentiment'].tolist(), index=word_counts.index)

# Creating a Word Cloud

all_words = ' '.join(word_counts['Word'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

# Plotting the Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Export Word Counts and Sentiment Scores to Excel with a specific style
output_excel_path = 'C:\\TestCode\\csat\\word_counts_and_sentiments.xlsx'

# Check if the file exists and remove it
if os.path.exists(output_excel_path):
    os.remove(output_excel_path)

# Use ExcelWriter to export the DataFrame with a specific style
with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
    word_counts.to_excel(writer, index=False, sheet_name='WordCountsAndSentiments', startrow=1, header=False)

    # Get the xlsxwriter workbook and worksheet objects
    workbook  = writer.book
    worksheet_word_counts = writer.sheets['WordCountsAndSentiments']

    # Get the dimensions of the DataFrame
    num_rows_wc, num_cols_wc = word_counts.shape

    # Create a list of column headers, to use in add_table()
    column_settings_wc = [{'header': column} for column in word_counts.columns]

    # Add the Excel table structure. Pandas will add the data
    worksheet_word_counts.add_table(0, 0, num_rows_wc, num_cols_wc - 1, {'columns': column_settings_wc})

print(f"Word counts and sentiment scores exported to {output_excel_path}")

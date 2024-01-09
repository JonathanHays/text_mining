import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

excel_file_path = 'C:\\TestCode\\csat\\sampleData.xlsx'

# Creating a DataFrame
df = pd.read_excel(excel_file_path)

# Tokenizing and removing stopwords
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return words

df['Tokenized'] = df['Responses'].apply(preprocess_text)

# Counting word occurrences
word_counts = {}
for tokens in df['Tokenized']:
    for word in tokens:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1

# Creating a Word Cloud
all_words = ' '.join([' '.join(tokens) for tokens in df['Tokenized']])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)

# Plotting the Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Export Word Counts to Excel with a specific style
output_excel_path = 'C:\\TestCode\\csat\\word_counts.xlsx'

# Create a DataFrame from word_counts
df_word_counts = pd.DataFrame(list(word_counts.items()), columns=['Word', 'Count'])

# Check if the file exists and remove it
if os.path.exists(output_excel_path):
    os.remove(output_excel_path)

# Use ExcelWriter to export the DataFrame with a specific style
with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
    df_word_counts.to_excel(writer, index=False, sheet_name='WordCounts', startrow=1, header=False)

    # Get the xlsxwriter workbook and worksheet objects
    workbook  = writer.book
    worksheet = writer.sheets['WordCounts']

    # Get the dimensions of the DataFrame
    num_rows, num_cols = df_word_counts.shape

    # Create a list of column headers, to use in add_table()
    column_settings = [{'header': column} for column in df_word_counts.columns]

    # Add the Excel table structure. Pandas will add the data
    worksheet.add_table(0, 0, num_rows, num_cols - 1, {'columns': column_settings})

print(f"Word counts exported to {output_excel_path}")

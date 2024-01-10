# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 17:46:50 2024

@author: jonha
"""

import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load your dataset
# Assuming your dataset has a 'Response' column with the text and a 'rating' column with sentiment scores
# Replace 'your_dataset.csv' with the actual file path or data loading method you are using
excel_file_path = 'C:\\TestCode\\csat\\sampleMachineData.xlsx'
df = pd.read_excel(excel_file_path)


# Convert sentiment scores to categories (negative, neutral, positive)
df['Sentiment'] = pd.cut(df['rating'], bins=[-float('inf'), 2, 3, float('inf')], labels=['negative', 'neutral', 'positive'])

# Convert timestamp to datetime (if applicable)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Group by date and sentiment to analyze trends
trend_data = df.groupby([df['Timestamp'].dt.date, 'Sentiment']).size().unstack().fillna(0)

# Plot the trend over time
trend_data.plot(kind='line', marker='o')
plt.title('Sentiment Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend(title='Sentiment')
plt.show()

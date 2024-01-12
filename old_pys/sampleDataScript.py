# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 18:53:05 2024

@author: jonha
"""

import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta

# Create a Faker instance
fake = Faker()

# Number of rows in the dataset
num_rows = 100000

# Generate random survey responses
responses = [fake.sentence() for _ in range(num_rows)]

# Generate random ratings (1-5)
ratings = [random.randint(1, 5) for _ in range(num_rows)]

# Generate random timestamps within a specific date range
start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 12, 31)
timestamps = [fake.date_time_between(start_date, end_date).strftime("%Y-%m-%d %H:%M:%S") for _ in range(num_rows)]

# Create a DataFrame
df = pd.DataFrame({'response': responses, 'rating': ratings, 'timestamp': timestamps})

# Specify the export path for the Excel file
export_path = 'C:\\TestCode\\csat\\excel\\sampleMachineData.xlsx'

# Save the DataFrame to an Excel file with the specified export path
df.to_excel(export_path, index=False, engine='openpyxl')

print(f"Excel file created successfully at: {export_path}")

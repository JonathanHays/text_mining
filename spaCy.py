import pandas as pd
import spacy
from collections import Counter

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Example DataFrame creation (replace this with your actual data)
excel_file_path = 'C:\TestCode\csat\sampleData.xlsx'

# Creating a DataFrame
df = pd.read_excel(excel_file_path)

# Process each response with SpaCy
df['SpacyDoc'] = df['Responses'].apply(nlp)

# Named Entity Recognition (NER) Example
entities = []
for doc in df['SpacyDoc']:
    entities.extend([(ent.text, ent.label_) for ent in doc.ents])

# Display Named Entities
print("Named Entities:")
for entity, label in entities:
    print(f"{entity} ({label})")

# Part-of-Speech (POS) Tagging and Word Counting Example
pos_tags = []
word_counts = Counter()

for doc in df['SpacyDoc']:
    pos_tags.extend([(token.text, token.pos_) for token in doc])
    for token in doc:
        if token.is_alpha and not token.is_stop:
            word_counts[token.text.lower()] += 1

# Display POS Tags
print("\nPart-of-Speech Tags:")
for token, pos in pos_tags:
    print(f"{token} ({pos})")

# Display Word Frequencies
print("\nWord Frequencies:")
for word, count in word_counts.items():
    print(f"{word}: {count}")


import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()

excel_file_path = "C:\\TestCode\\csat\\excel\\cannedResponses.xlsx"

# Read the Excel file into a DataFrame
df = pd.read_excel(excel_file_path)


def generate_unique_id(existing_ids):
    new_id = fake.uuid4()
    while new_id in existing_ids:
        new_id = fake.uuid4()
    existing_ids.add(new_id)
    return new_id

def generate_random_agent(existing_agent_ids):
    agent_id = generate_unique_id(existing_agent_ids)
    first_name = fake.first_name()
    last_name = fake.last_name()
    return {'agent_id': agent_id, 'first_name': first_name, 'last_name': last_name}



def generate_fake_data(num_tickets):
    # Create a set for ticket IDs and agent IDs to ensure uniqueness
    generated_ticket_ids = set()
    generated_agent_ids = set()

    # Create a list to store generated data
    ticket_data = []
    
    chat_agents = []
    
    for _ in range(30):
        agent = generate_random_agent(generated_agent_ids)
        chat_agents.append(agent)
    
    for _ in range(num_tickets):
        # Generate a unique chat_transcript_id
        chat_transcript_id = generate_unique_id(generated_ticket_ids)
        chat_created_datetime = fake.date_time_between(start_date='-1y', end_date='now')
        chat_resolved_datetime = chat_created_datetime + timedelta(hours=random.randint(1, 72))
        chat_first_reply_datetime = chat_created_datetime + timedelta(minutes=random.randint(5, 60))
        chat_reason = fake.random_element(elements=('Technical Issue', 'Billing Inquiry', 'Product Inquiry', 'Other'))
        
        # Assign an agent for each month
        i = random.randint(0, len(chat_agents) - 1)
        chat_agent_id = chat_agents[i]['agent_id']
        agent_name = f"{chat_agents[i]['first_name']} {chat_agents[i]['last_name']}"
        
        chat_survey_id = fake.uuid4() if random.random() < 0.8 else None  # 80% chance of having a survey
        random_row = df.sample(n=1)
        chat_survey_rating = random_row['rating'].values[0] if chat_survey_id else None
        chat_survey_response = random_row['response'].values[0] if chat_survey_id and random.random() < 0.88 else None

        # Create a dictionary for the ticket and append to the list
        ticket = {
            'chat_transcript_id': chat_transcript_id,
            'chat_created_datetime': chat_created_datetime,
            'chat_resolved_datetime': chat_resolved_datetime,
            'chat_first_reply_datetime': chat_first_reply_datetime,
            'chat_reason': chat_reason,
            'chat_agent_id': chat_agent_id,
            'agent_name': agent_name,
            'chat_survey_id': chat_survey_id,
            'chat_survey_rating': chat_survey_rating,
            'chat_survey_response': chat_survey_response
        }

        ticket_data.append(ticket)

    return ticket_data

# Set the number of fake tickets you want to generate
num_fake_tickets = 100000

# Generate fake ticket data
fake_tickets = generate_fake_data(num_fake_tickets)

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(fake_tickets)

# Save the DataFrame to an Excel file
excel_file_path = "C:\\TestCode\\csat\\excel\\sampleCustomerChatData.xlsx"
df.to_excel(excel_file_path, index=False)

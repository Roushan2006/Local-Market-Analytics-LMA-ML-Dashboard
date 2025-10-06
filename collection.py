import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta


NUM_BUSINESSES = 10000
NUM_USERS = 10000
NUM_INTERACTIONS = 10000 


fake = Faker()


CATEGORIES = [
    'Coffee Shop', 'Restaurant', 'Gym', 'Bookstore', 'Plumber', 
    'Boutique', 'Hair Salon', 'Pet Store', 'Auto Repair', 'Bakery'
]
CITIES = ['Saharsa', 'Khagaria', 'Mansi', 'Begusarai', 'Meerut']
INCOME_BRACKETS = ['Low', 'Medium', 'High']
VISIT_FREQUENCIES = ['Daily', 'Weekly', 'Monthly', 'Rarely']
SEGMENT_TARGETS = ['Low Value', 'Medium Value', 'High Value']
INTERACTION_TYPES = ['view', 'click', 'purchase', 'review', 'visit']


# =======================================================
# 1. Generate Businesses Table (10,000 Rows)
# =======================================================

business_data = {
    'business_id': range(10001, 10001 + NUM_BUSINESSES),
    'category': [random.choice(CATEGORIES) for _ in range(NUM_BUSINESSES)],
    'city': [random.choice(CITIES) for _ in range(NUM_BUSINESSES)],
    'price_range': np.random.randint(1, 5, NUM_BUSINESSES), # 1 to 4
    'avg_rating': np.clip(np.random.normal(4.1, 0.5, NUM_BUSINESSES), 2.0, 5.0).round(1),
    'review_count': np.random.randint(10, 5000, NUM_BUSINESSES),
    'promo_activity': np.random.choice([0, 1], NUM_BUSINESSES, p=[0.6, 0.4]),
}

df_businesses = pd.DataFrame(business_data)


base_sales = df_businesses['review_count'] * 15 + df_businesses['avg_rating'] * 10000
promo_boost = df_businesses['promo_activity'] * 5000 
noise = np.random.normal(0, 10000, NUM_BUSINESSES)

df_businesses['sales_target'] = (base_sales + promo_boost + noise).round(2)
df_businesses['sales_target'] = np.clip(df_businesses['sales_target'], 5000, 1000000) # Ensure positive and bounded sales


# =======================================================
# 2. Generate Users Table (10,000 Rows)
# =======================================================

user_data = {
    'user_id': range(50001, 50001 + NUM_USERS),
    'age': np.random.randint(18, 70, NUM_USERS),
    'income_bracket': np.random.choice(INCOME_BRACKETS, NUM_USERS, p=[0.3, 0.5, 0.2]),
    'pref_category_1': [random.choice(CATEGORIES) for _ in range(NUM_USERS)],
    'avg_spend_3mo': np.clip(np.random.normal(40, 30, NUM_USERS), 5, 200).round(2),
    'visit_frequency': np.random.choice(VISIT_FREQUENCIES, NUM_USERS, p=[0.15, 0.35, 0.3, 0.2]),
}

df_users = pd.DataFrame(user_data)


def assign_segment(row):
    if row['avg_spend_3mo'] > 80 and row['visit_frequency'] in ['Daily', 'Weekly']:
        return 'High Value'
    elif row['avg_spend_3mo'] > 30 and row['visit_frequency'] in ['Weekly', 'Monthly']:
        return 'Medium Value'
    else:
        return 'Low Value'

df_users['segment_target'] = df_users.apply(assign_segment, axis=1)


# =======================================================
# 3. Generate Interactions Table (10,000 Rows)
# =======================================================

random_user_ids = np.random.choice(df_users['user_id'], NUM_INTERACTIONS)
random_business_ids = np.random.choice(df_businesses['business_id'], NUM_INTERACTIONS)

start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 1, 1)

interaction_data = {
    'interaction_id': range(900001, 900001 + NUM_INTERACTIONS),
    'user_id': random_user_ids,
    'business_id': random_business_ids,
    'timestamp': [start_date + timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds()))) for _ in range(NUM_INTERACTIONS)],
    'distance_km': np.clip(np.random.normal(5, 5, NUM_INTERACTIONS), 0.1, 50).round(1),
    'interaction_type': np.random.choice(INTERACTION_TYPES, NUM_INTERACTIONS, p=[0.4, 0.2, 0.15, 0.05, 0.2]),
    'time_spent_sec': np.random.randint(5, 600, NUM_INTERACTIONS),
}

df_interactions = pd.DataFrame(interaction_data)


def assign_conversion(row):
    if row['interaction_type'] in ['purchase', 'visit']:
       
        return 1
    elif row['interaction_type'] in ['click', 'review']:

        prob_convert = 0.5 - (row['distance_km'] / 100) - (row['time_spent_sec'] / 1000)
        return 1 if random.random() < max(0.05, prob_convert) else 0 
    else: 
        prob_convert = 0.2 - (row['distance_km'] / 200) - (row['time_spent_sec'] / 1500)
        return 1 if random.random() < max(0.01, prob_convert) else 0

df_interactions['conversion_target'] = df_interactions.apply(assign_conversion, axis=1)


# =======================================================
# 4. Save to CSV Files
# =======================================================

df_businesses.to_csv('local_businesses.csv', index=False)
df_users.to_csv('local_users.csv', index=False)
df_interactions.to_csv('user_interactions.csv', index=False)

print(f"Successfully generated {NUM_BUSINESSES} business records.")
print(f"Successfully generated {NUM_USERS} user records.")
print(f"Successfully generated {NUM_INTERACTIONS} interaction records.")
print("\nFiles saved: local_businesses.csv, local_users.csv, user_interactions.csv")
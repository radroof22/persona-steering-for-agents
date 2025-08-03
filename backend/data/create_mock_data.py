import pandas as pd
import numpy as np
from typing import List, Dict


def create_mock_users() -> pd.DataFrame:
    """Create mock users with diverse descriptions for personalization testing."""
    users_data = [
        {"user_id": "user_001", "description": "A formal business professional who prefers concise, professional language"},
        {"user_id": "user_002", "description": "A casual tech enthusiast who uses informal language and tech jargon"},
        {"user_id": "user_003", "description": "An academic researcher who prefers detailed, analytical explanations"},
        {"user_id": "user_004", "description": "A creative writer who enjoys flowery, descriptive language"},
        {"user_id": "user_005", "description": "A student who prefers simple, straightforward explanations"},
        {"user_id": "user_006", "description": "A senior executive who values brevity and actionable insights"},
        {"user_id": "user_007", "description": "A marketing professional who uses persuasive, engaging language"},
        {"user_id": "user_008", "description": "A technical consultant who prefers structured, step-by-step explanations"},
        {"user_id": "user_009", "description": "A customer service representative who uses friendly, helpful language"},
        {"user_id": "user_010", "description": "A data scientist who prefers precise, quantitative language"}
    ]
    return pd.DataFrame(users_data)


def create_mock_queries() -> pd.DataFrame:
    """Create mock queries that can be clustered together."""
    queries_data = [
        # Cluster 1: Weather-related queries
        {"user_id": "user_001", "question": "What's the weather like today?"},
        {"user_id": "user_002", "question": "How's the weather looking?"},
        {"user_id": "user_003", "question": "What are the current weather conditions?"},
        {"user_id": "user_004", "question": "Can you tell me about today's weather?"},
        
        # Cluster 2: Restaurant recommendations
        {"user_id": "user_005", "question": "Where should I eat tonight?"},
        {"user_id": "user_006", "question": "Can you recommend a good restaurant?"},
        {"user_id": "user_007", "question": "What restaurants are nearby?"},
        {"user_id": "user_008", "question": "I'm looking for a place to dine"},
        
        # Cluster 3: Technical questions
        {"user_id": "user_009", "question": "How do I install Python?"},
        {"user_id": "user_010", "question": "What's the best way to set up Python?"},
        {"user_id": "user_001", "question": "Can you help me install Python on my computer?"},
        
        # Cluster 4: Travel questions
        {"user_id": "user_002", "question": "What's the best time to visit Paris?"},
        {"user_id": "user_003", "question": "When should I plan a trip to Paris?"},
        {"user_id": "user_004", "question": "What's the ideal season for visiting Paris?"},
        
        # Cluster 5: Health questions
        {"user_id": "user_005", "question": "How can I improve my sleep?"},
        {"user_id": "user_006", "question": "What are some tips for better sleep?"},
        {"user_id": "user_007", "question": "How do I get better sleep quality?"},
        
        # Some singleton clusters
        {"user_id": "user_008", "question": "What's the capital of Mongolia?"},
        {"user_id": "user_009", "question": "How do quantum computers work?"},
        {"user_id": "user_010", "question": "What's the recipe for sourdough bread?"}
    ]
    return pd.DataFrame(queries_data)


def main():
    """Generate and save mock data as parquet files."""
    print("Creating mock users...")
    users_df = create_mock_users()
    users_df.to_parquet("app/data/mock_users.parquet", index=False)
    print(f"Created {len(users_df)} mock users")
    
    print("Creating mock queries...")
    queries_df = create_mock_queries()
    queries_df.to_parquet("app/data/mock_queries.parquet", index=False)
    print(f"Created {len(queries_df)} mock queries")
    
    print("Mock data generation complete!")


if __name__ == "__main__":
    main() 
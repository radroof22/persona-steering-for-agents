import pandas as pd
import os
from datetime import datetime
import uuid
from typing import List, Dict, Any

class ParquetDatabase:
    """Database class for handling Parquet file operations"""
    
    def __init__(self, users_file: str = "users.parquet", queries_file: str = "queries.parquet", answers_file: str = "answers.parquet"):
        self.users_file = users_file
        self.queries_file = queries_file
        self.answers_file = answers_file
        self._ensure_files_exist()
    
    def _ensure_files_exist(self):
        """Ensure the parquet database files exist with proper schema"""
        if not os.path.exists(self.users_file):
            empty_users_df = pd.DataFrame(columns=[
                'unique_id', 'user_id', 'sample_text_corpus', 
                'natural_language_description', 'registration_date'
            ])
            empty_users_df.to_parquet(self.users_file, index=False)
        
        if not os.path.exists(self.queries_file):
            empty_queries_df = pd.DataFrame(columns=[
                'query_id', 'query_text', 'timestamp', 'cluster'
            ])
            empty_queries_df.to_parquet(self.queries_file, index=False)
        
        if not os.path.exists(self.answers_file):
            empty_answers_df = pd.DataFrame(columns=[
                'answer_id', 'query_text', 'answer_text', 'user_id', 'timestamp'
            ])
            empty_answers_df.to_parquet(self.answers_file, index=False)
    
    def load_users(self) -> pd.DataFrame:
        """Load users from parquet file"""
        self._ensure_files_exist()
        try:
            return pd.read_parquet(self.users_file)
        except Exception as e:
            # If the file is corrupted, recreate it
            print(f"Warning: Could not read {self.users_file}, recreating: {e}")
            empty_df = pd.DataFrame(columns=['unique_id', 'user_id', 'sample_text_corpus', 'natural_language_description', 'registration_date'])
            empty_df.to_parquet(self.users_file, index=False)
            return empty_df
    
    def save_users(self, df: pd.DataFrame):
        """Save users to parquet file"""
        df.to_parquet(self.users_file, index=False)
    
    def load_queries(self) -> pd.DataFrame:
        """Load queries from parquet file"""
        self._ensure_files_exist()
        try:
            return pd.read_parquet(self.queries_file)
        except Exception as e:
            # If the file is corrupted, recreate it
            print(f"Warning: Could not read {self.queries_file}, recreating: {e}")
            empty_df = pd.DataFrame(columns=['query_id', 'query_text', 'timestamp', 'cluster'])
            empty_df.to_parquet(self.queries_file, index=False)
            return empty_df
    
    def save_queries(self, df: pd.DataFrame):
        """Save queries to parquet file"""
        df.to_parquet(self.queries_file, index=False)
    
    def load_answers(self) -> pd.DataFrame:
        """Load answers from parquet file"""
        self._ensure_files_exist()
        try:
            return pd.read_parquet(self.answers_file)
        except Exception as e:
            # If the file is corrupted, recreate it
            print(f"Warning: Could not read {self.answers_file}, recreating: {e}")
            empty_df = pd.DataFrame(columns=['answer_id', 'query_text', 'answer_text', 'user_id', 'timestamp'])
            empty_df.to_parquet(self.answers_file, index=False)
            return empty_df
    
    def save_answers(self, df: pd.DataFrame):
        """Save answers to parquet file"""
        df.to_parquet(self.answers_file, index=False)
    
    def add_user(self, user_id: str, sample_text_corpus: str = None, 
                 natural_language_description: str = None) -> Dict[str, Any]:
        """Add a new user to the database"""
        users_df = self.load_users()
        
        # Check if user already exists
        if user_id in users_df['user_id'].values:
            raise ValueError(f"User {user_id} already exists")
        
        # Create new user record
        new_user = {
            'unique_id': str(uuid.uuid4()),
            'user_id': user_id,
            'sample_text_corpus': sample_text_corpus,
            'natural_language_description': natural_language_description,
            'registration_date': datetime.now().isoformat()
        }
        
        # Add to DataFrame
        users_df = pd.concat([users_df, pd.DataFrame([new_user])], ignore_index=True)
        self.save_users(users_df)
        
        return new_user
    
    def add_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Add new queries to the database"""
        queries_df = self.load_queries()
        
        new_queries = []
        for query in queries:
            new_query = {
                'query_id': str(uuid.uuid4()),
                'query_text': query,
                'timestamp': datetime.now().isoformat(),
                'cluster': None  # Will be updated by clustering
            }
            new_queries.append(new_query)
        
        # Add to DataFrame
        new_queries_df = pd.DataFrame(new_queries)
        queries_df = pd.concat([queries_df, new_queries_df], ignore_index=True)
        self.save_queries(queries_df)
        
        return new_queries
    
    def add_answer(self, query: str, answer: str, user_id: str = None) -> Dict[str, Any]:
        """Add a new query-answer pair to the database"""
        answers_df = self.load_answers()
        
        new_answer = {
            'answer_id': str(uuid.uuid4()),
            'query_text': query,
            'answer_text': answer,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to DataFrame
        answers_df = pd.concat([answers_df, pd.DataFrame([new_answer])], ignore_index=True)
        self.save_answers(answers_df)
        
        return new_answer
    
    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users as a list of dictionaries"""
        users_df = self.load_users()
        return users_df.to_dict('records')
    
    def get_all_queries(self) -> List[Dict[str, Any]]:
        """Get all queries as a list of dictionaries"""
        queries_df = self.load_queries()
        return queries_df.to_dict('records')
    
    def get_all_answers(self) -> List[Dict[str, Any]]:
        """Get all answers as a list of dictionaries"""
        answers_df = self.load_answers()
        return answers_df.to_dict('records')
    
    def get_query_texts(self) -> List[str]:
        """Get all query texts as a list"""
        queries_df = self.load_queries()
        return queries_df['query_text'].tolist()
    
    def reset_database_files(self):
        """Reset all database files to empty state (useful for testing)"""
        try:
            # Reset users
            empty_users_df = pd.DataFrame(columns=[
                'unique_id', 'user_id', 'sample_text_corpus', 
                'natural_language_description', 'registration_date'
            ])
            empty_users_df.to_parquet(self.users_file, index=False)
            
            # Reset queries
            empty_queries_df = pd.DataFrame(columns=[
                'query_id', 'query_text', 'timestamp', 'cluster'
            ])
            empty_queries_df.to_parquet(self.queries_file, index=False)
            
            # Reset answers
            empty_answers_df = pd.DataFrame(columns=[
                'answer_id', 'query_text', 'answer_text', 'user_id', 'timestamp'
            ])
            empty_answers_df.to_parquet(self.answers_file, index=False)
            
            print("✅ Database files reset successfully")
        except Exception as e:
            print(f"❌ Error resetting database files: {e}")
            raise 
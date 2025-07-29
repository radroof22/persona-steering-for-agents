# Sentosa

| LLMs are expensive for individual users. Bundle users together to make it cheaper, but still personalized.


# Modes

## Rewrite

If you currently have a lot of data on your users, you can use `rewrite` mode. This will attempt to rewrite the response in text the user is already familiar with

## Prompt

Given a natural language description of your user (request about me about user), the response will be rewritten using a smaller LLM for the user.


## Finetuning

Given a corpus of the users text, the response will be finetuned to something the model believes the users is better.

## Query Answering

Use T5 models to answer user queries with optional personalization based on user style preferences.

# Setup

## Environment Setup

This project uses conda/mamba for environment management. Install dependencies using:

```bash
# Using mamba (recommended)
mamba env create -f environment.yml

# Or using conda
conda env create -f environment.yml

# Activate the environment
mamba activate sentosa
# or
conda activate sentosa
```

The environment includes all necessary dependencies:
- **Core**: Python 3.11, FastAPI, Uvicorn
- **ML/AI**: PyTorch, Transformers, Scikit-learn
- **Data**: Pandas, NumPy, PyArrow
- **Testing**: Pytest, Requests
- **Other**: Pydantic, Accelerate

### Environment Check

Before running the application, check if your environment is properly set up:

```bash
python check_environment.py
```

This will verify:
- You're in the correct conda environment
- All required dependencies are installed
- Database functionality works correctly

## API Endpoints

### User Registration
- **POST** `/register` - Register a new user with optional sample text corpus and natural language description
- **GET** `/users` - Get all registered users

### Query Processing
- **POST** `/batch-queries` - Process a batch of queries and identify clusters using DBSCAN (clusters only the submitted queries)
- **GET** `/queries` - Get all stored queries

### Query Answering
- **POST** `/answer-query` - Answer queries using T5 model with optional user personalization
- **GET** `/answers` - Get all stored query-answer pairs

### Clustering Configuration
- **POST** `/configure-clustering` - Configure clustering parameters (epsilon, min_samples, etc.)
- **GET** `/clustering-stats` - Get clustering statistics and epsilon recommendations

### Response Rewriting
- **POST** `/rewrite` - Rewrite responses to match user's familiar writing style using Hugging Face models
- **GET** `/style-analysis/{user_id}` - Get detailed analysis of user's writing style

### Data Storage
The application uses Parquet files as the database:
- `users.parquet` - Stores user registration data
- `queries.parquet` - Stores query data with clustering results
- `answers.parquet` - Stores query-answer pairs

### Model Requirements
The application uses Hugging Face models:
- **Primary**: T5-base for text generation, style adaptation, and query answering
- **Fallback**: T5-small if the primary model fails to load
- **GPU Support**: Automatically uses GPU if available for faster processing

## Usage Examples

### Query Answering

```python
import requests

# Basic query answering
response = requests.post("http://localhost:8000/answer-query", json={
    "query": "What is machine learning?"
})

# Personalized query answering
response = requests.post("http://localhost:8000/answer-query", json={
    "query": "How do neural networks work?",
    "user_id": "bob_writer"
})
```

### Testing

Run the comprehensive test suite:
```bash
# Run all tests
pytest test_api.py -v

# Run specific test classes
pytest test_api.py::TestQueryAnswering -v
pytest test_api.py::TestUserRegistration -v

# Run with output
pytest test_api.py -v -s
```

Run the simple query API test:
```bash
pytest test_query_api.py -v -s
```

Use the test runner script for easy testing:
```bash
# Quick test (query API only)
python run_tests.py --test-type quick

# Comprehensive test suite
python run_tests.py --test-type comprehensive -v -s

# All tests
python run_tests.py --test-type all -v

# Help
python run_tests.py --help
```

### Database Management

If you encounter issues with corrupted parquet files, you can reset the database:

```bash
# Reset all database files to empty state
python reset_database.py
```

This will recreate all parquet files with the correct schema and empty data.

### Test Structure

The test suite is organized into logical test classes:

- **TestAPIHealth**: Basic API connectivity tests, dependency checks, and database file validation
- **TestUserRegistration**: User registration and management tests
- **TestQueryProcessing**: Query clustering and processing tests
- **TestClusteringConfiguration**: Clustering parameter configuration tests
- **TestResponseRewriting**: Response rewriting and style analysis tests
- **TestQueryAnswering**: Query answering functionality tests
- **TestDataRetrieval**: Data retrieval endpoint tests
- **TestAllQueryBatches**: Comprehensive query batch processing tests

Each test class includes proper error handling, timeouts, and graceful skipping when services are unavailable.
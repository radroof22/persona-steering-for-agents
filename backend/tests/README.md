# Backend Service Tests

This directory contains comprehensive tests for the backend services.

## Test Structure

### `test_clusterer.py`

Contains three main test classes:

1. **`TestQueryClusterer`** - Unit tests for the QueryClusterer class
   - Tests initialization, model loading, encoding, and clustering
   - **Primary focus**: Testing `process_queries` function by mocking `cluster_queries`
   - Covers various scenarios: normal clusters, noise points, mixed cases, edge cases

2. **`TestCreateClusterer`** - Tests for the factory function
   - Tests default and custom parameter creation

3. **`TestClustererIntegration`** - Integration tests
   - Tests the full clustering pipeline without mocking

### `test_rewriter.py`

Contains four main test classes:

1. **`TestStylePersonalizer`** - Unit tests for the StylePersonalizer class
   - Tests initialization, prompt creation, and personalization logic
   - **Primary focus**: Testing `create_personalized_rewrites` function by mocking `personalize_query`
   - Covers various scenarios: normal cases, missing descriptions, empty input, exceptions

2. **`TestPersonalizeQuery`** - Tests for the personalize_query method
   - Tests successful personalization and exception handling

3. **`TestPersonalizeResponse`** - Tests for the personalize_response method
   - Tests response personalization functionality

4. **`TestCreateStylePersonalizer`** - Tests for the factory function
   - Tests default and custom parameter creation

## Key Test Scenarios

### `process_queries` Function Tests (Primary Focus)

1. **Normal Clusters**: Tests when all queries form proper clusters
2. **Noise Points**: Tests handling of DBSCAN noise points (label -1) as singleton clusters
3. **All Noise**: Tests when all queries are noise points
4. **Empty Input**: Tests edge case with empty query list
5. **Single Query**: Tests single query handling
6. **Mixed Cases**: Tests combination of normal clusters and noise points

### `create_personalized_rewrites` Function Tests (Primary Focus)

1. **Normal Case**: Tests with complete user descriptions and successful personalization
2. **Missing Descriptions**: Tests handling of missing user descriptions (uses default)
3. **Empty Input**: Tests edge case with empty query lists
4. **Single Query**: Tests single query personalization
5. **Mismatched Lengths**: Tests when queries and rewritten queries have different lengths
6. **Exception Handling**: Tests fallback behavior when personalization fails

### Mocking Strategy

**Clusterer Tests**: Use `@patch.object(QueryClusterer, 'cluster_queries')` to mock the clustering algorithm, allowing us to:
- Test the `process_queries` logic independently
- Control cluster assignments for different scenarios
- Avoid expensive embedding and clustering operations during testing
- Focus on the business logic of grouping queries and handling noise points

**Rewriter Tests**: Use `@patch.object(StylePersonalizer, 'personalize_query')` to mock the personalization function, allowing us to:
- Test the `create_personalized_rewrites` logic independently
- Control personalization results for different scenarios
- Avoid expensive LLM calls during testing
- Focus on the business logic of creating PersonalizedRewrite objects

## Running Tests

### Prerequisites

Install test dependencies:
```bash
pip install -r requirements-test.txt
```

### Run All Tests
```bash
# From backend directory
./run_tests.sh

# Or directly with pytest
python -m pytest tests/test_clusterer.py -v
```

### Run Specific Test Classes
```bash
# Clusterer tests
python -m pytest tests/test_clusterer.py::TestQueryClusterer -v
python -m pytest tests/test_clusterer.py::TestCreateClusterer -v
python -m pytest tests/test_clusterer.py::TestClustererIntegration -v

# Rewriter tests
python -m pytest tests/test_rewriter.py::TestStylePersonalizer -v
python -m pytest tests/test_rewriter.py::TestPersonalizeQuery -v
python -m pytest tests/test_rewriter.py::TestPersonalizeResponse -v
python -m pytest tests/test_rewriter.py::TestCreateStylePersonalizer -v
```

### Run with Coverage
```bash
coverage run -m pytest tests/test_clusterer.py
coverage report
coverage html  # Generate HTML report
```

## Test Data

The tests use realistic sample queries covering common scenarios:
- Weather-related queries (similar semantic meaning)
- Password reset queries (similar intent)
- Store hours queries (unique queries)

This allows testing of both clustering behavior and noise point handling.

## Expected Behavior

### Normal Clustering
- Similar queries should be grouped together
- Each cluster gets a unique positive ID
- User IDs and queries are properly associated

### Noise Point Handling
- DBSCAN noise points (label -1) become singleton clusters
- Each noise point gets a unique positive ID
- Singleton clusters maintain proper user ID and query associations

### Edge Cases
- Empty input returns empty list
- Single query becomes single cluster
- Mixed normal/noise scenarios handled correctly 
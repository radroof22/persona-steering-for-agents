# Frontend Demo for Clustered Prompt Personalization

A Streamlit-based demo frontend for the Personalized Query Rewriting API. This application allows users to upload CSV files with prompts, provide personal descriptions, and visualize how the system clusters and personalizes their queries.

## Features

- 📁 CSV file upload with validation
- 👤 User profile input for personalization
- 🎯 Clustered prompt visualization
- ✨ Personalized rewrite display
- 💾 Results download as CSV
- 🔧 Developer output toggle for debugging

## Setup

### Prerequisites

- Python 3.8+
- Backend API running on `http://localhost:8000`

### Installation

1. Ensure you have the sentosa conda environment activated:
   ```bash
   conda activate sentosa
   ```

2. (Optional) Create a `.env` file in the frontend directory to customize API URL:
   ```
   API_URL=http://localhost:8000
   ```

## Usage

### Running the Application

1. Ensure the backend API is running:
   ```bash
   # From the root directory
   cd app
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. Start the Streamlit frontend:
   ```bash
   # From the frontend directory
   streamlit run main.py
   ```

3. Open your browser to `http://localhost:8501`

### Data Format

The demo requires two CSV files:

#### 1. Users CSV (`sample_users.csv`)
```csv
user_id,description
user_001,I'm a data scientist with 5 years of experience in machine learning...
user_002,I'm a software engineer transitioning into ML...
```

**Requirements:**
- Must contain columns: `user_id`, `description`
- No empty values in either column
- No duplicate `user_id` values

#### 2. Queries CSV (`sample_queries.csv`)
```csv
user_id,question
user_001,How do I improve my resume for data science roles?
user_002,What are attention heads in transformer models?
```

**Requirements:**
- Must contain columns: `user_id`, `question`
- No empty values in either column
- `user_id` values must exist in the users CSV

### Workflow

1. **Landing Page**: Upload users CSV and queries CSV files
2. **Processing**: The system clusters queries and generates personalized rewrites for each user
3. **Results Page**: View clustered prompts with original and rewritten versions
4. **Download**: Export results as CSV for further analysis

## Architecture

```
frontend/
├── main.py              # Main Streamlit application
├── api_client.py        # HTTP client for API communication
├── utils.py             # Data processing and validation utilities
├── config.py            # Configuration and styling
└── README.md           # This file
```

## API Integration

The frontend communicates with the backend API endpoints:

- `GET /health` - Health check
- `POST /generate` - Generate clustered and personalized rewrites
- `POST /clusters` - Get cluster information only

## Customization

### Styling

Modify colors and styling in `config.py`:

```python
COLORS = {
    "background": "#FAFAFA",
    "primary": "#007ACC",
    "secondary": "#00BFA5",
    "text": "#333333"
}
```

### API Configuration

Change the API URL in `config.py` or set the `API_URL` environment variable.

## Troubleshooting

### Common Issues

1. **API Connection Error**: Ensure the backend is running on the correct port
2. **CSV Upload Error**: Check that your CSV has a `query` column
3. **Processing Timeout**: Large files may take longer to process

### Debug Mode

Enable developer output in the results page to see:
- Raw API responses
- Cluster metadata
- Error details

## Future Enhancements

- Additional rewrite strategies
- Clustering parameter tuning
- Embedding visualization
- User profile upload support
- Real-time processing status 
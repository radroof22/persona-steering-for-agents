Product Requirements Document (PRD): Demo Frontend for Query Clustering & Personalization System
📌 Overview
This is a demo frontend for a system that clusters user-submitted queries and rewrites them based on user profile descriptions using an LLM. It’s designed to showcase the clustering and rewriting pipeline, allowing users to upload their own CSV data and view rewritten results grouped by cluster.

The demo is intended for technical stakeholders and should be easy to use, inspect, and extend.

🏗️ Stack & Architecture
Frontend Framework: Streamlit

Backend API: FastAPI (already implemented)

Data Input: CSV upload from user (format specified below)

Interface: Multi-page Streamlit app (main.py)

Communication: Use httpx or requests to connect to FastAPI endpoints

🎨 Visual Design
Color Palette
Background: #FAFAFA

Primary accent: #007ACC (Azure Blue)

Secondary accent: #00BFA5 (Turquoise)

Font color: #333333

Font: System default sans-serif

Copy and Tone
Use friendly but precise language. Examples:

“Upload your prompts to see how we cluster and rewrite them.”

“We’ll personalize them based on your profile.”

“You can inspect the rewritten results side-by-side.”

📄 Pages & UX Flow
1. Landing Page (main.py)
Purpose:
Collect user inputs and trigger clustering + rewriting.

Elements:
Title: “🧠 Clustered Prompt Personalization Demo”

Subheader: “Visualize how we group and rewrite prompts based on your persona.”

Text area: "Tell us a bit about yourself" → stored as user_description

File uploader: "Upload your prompts as a CSV file"

Required format: CSV with one column named "query"

Dropdown: Rewrite strategy (default: "LLM - Reprompted"; others disabled for now)

Submit button: Runs the entire pipeline

2. Results Page
Purpose:
Display clustered queries with personalized rewrites.

Elements:
Cluster display:

Grouped by cluster ID

Each cluster in an st.expander, with heading like: Cluster 0 (3 prompts)

Each query:

Original: "Original: How do I improve my resume?"

Rewritten: "Rewritten: How can I enhance my resume for data science roles?"

Download button:

"Download results as CSV" (columns: cluster_id, original_query, rewritten_query, strategy)

Optional: Include “raw API output” toggle for advanced users

3. Developer View (Optional Toggle)
If a checkbox like "Show developer output" is selected:

Show raw JSON response from:

/cluster

/rewrite

Debug section with logs

Display internal IDs and embeddings (if available from API)

🔌 API Contracts
/cluster
POST

json
Copy
Edit
{
  "queries": ["query A", "query B", ...],
  "user_id": "user-001"
}
Response

json
Copy
Edit
[
  {
    "cluster_id": 0,
    "query": "query A",
    "user_id": "user-001"
  },
  ...
]
/rewrite
POST

json
Copy
Edit
{
  "queries": [...],
  "user_description": "I’m an ML researcher interested in LLM safety",
  "strategy": "reprompting"
}
Response

json
Copy
Edit
[
  {
    "original": "query A",
    "rewritten": "query A rewritten"
  },
  ...
]
🧪 CSV Format Specification
Required structure:
Uploaded file must be a .csv file with the following column:

query
How do I tune my model?
What are attention heads?

The system reads this column into a list of strings

Validation should show an error if the column "query" is not found

🔄 Future Extensions
The demo should be easy to extend with:

Additional rewrite strategies (rule-based, fine-tuned, corpus-aware)

Clustering parameter tuning (e.g., eps for DBSCAN)

Embedding visualization (e.g., 2D projection with plotly)

Optional profile upload: Add "profile.csv" support (e.g., user preferences)

🧪 Testing Instructions
To test locally:

bash
Copy
Edit
streamlit run main.py
Ensure .env contains API_URL=http://localhost:8000 or similar.


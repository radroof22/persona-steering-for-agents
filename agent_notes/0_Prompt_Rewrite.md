Personalized Query Rewriting with Clustering and Prompt-Based LLM
Overview
This project implements a proof-of-concept system that clusters similar user queries, rewrites one using a prompt-based LLM (from Hugging Face), and personalizes the rewritten query using user-specific style or context. This allows us to minimize LLM usage and cost by batching similar queries.

Folder Structure
graphql
Copy
Edit
app/
├── main.py                 # FastAPI entrypoint
├── models/
│   └── schema.py           # Pydantic models
├── services/
│   ├── clusterer.py        # Clustering logic
│   ├── llm_provider.py     # Hugging Face LLM access
│   ├── rewriter.py         # Style personalization logic
├── data/
│   ├── mock_users.parquet  # Mock users with user_id and description
│   └── mock_queries.parquet# Mock queries with user_id and question
├── utils/
│   └── loader.py           # Parquet loading / validation
tests/
├── test_api.py
├── test_clusterer.py
├── test_rewriter.py
Step-by-Step Implementation Plan
Step 1: Define Data Models
models/schema.py

User: user_id, description

Query: user_id, question

Step 2: Load and Validate Mock Data
utils/loader.py

Load mock_users.parquet and mock_queries.parquet

Merge them on user_id and validate format

Step 3: Implement Clustering Logic
services/clusterer.py

Use SentenceTransformers/all-MiniLM-L6-v2 to encode queries

Use DBSCAN to form clusters (doesn't assume a fixed k)

Group queries by cluster id

Step 4: Choose a Representative Query per Cluster
Heuristic: pick the query closest to the centroid

Create mapping of representative → [list of user_ids]

Step 5: Rewriting Representative Query
services/llm_provider.py

Use NousResearch/Hermes-2-Pro-Mistral-7B from Hugging Face

Prompt with the cluster query and high-level instruction

Step 6: Style Personalization
services/rewriter.py

For each user, rewrite the LLM output with a prompt like:

"Rewrite the following in the tone/style of the user described as: {user_description}"

Step 7: API Server
main.py (FastAPI)

Expose endpoint: /generate to return:

clustered queries

rewritten representative

personalized rewrites per user

Testing Plan (Pytest)
test_loader.py: test loading and schema checks for parquet files

test_clusterer.py: test DBSCAN logic to confirm:

Clusters are non-empty

At least one singleton cluster

At least one multi-user cluster

test_rewriter.py: test if personalized rewrites include relevant user descriptions

test_api.py: test FastAPI endpoint returns expected JSON

Bonus Notes
DBSCAN was chosen for flexibility in cluster sizes.

Using .parquet allows seamless scaling to databases like DuckDB or Postgres later.

Diffusion models (e.g., Stable Diffusion for Text) may be able to generate stylistically coherent rewrites, but aren’t ideal for this phase due to slower text alignment. May revisit later.

Next Steps
🧪 Finetuning
Fine-tune the rewrite model on a dataset of user queries + preferred rewrites

Helps LLM model adapt better to rewriting tasks with user-specific nuances

🌀 Diffusion Rewriting (Future Work)
Explore using latent diffusion transformers to rewrite text using image-conditioned or noise-to-sequence models

May allow richer personalization at lower latency once inference is optimized

⚙️ Rule-Based Rewriting
Augment the rewriter with rule-based logic for known user patterns (e.g., “always use formal tone for user_id = X”)

Could reduce reliance on LLM calls further in production


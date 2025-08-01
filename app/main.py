from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import List

from app.models.schema import Query, GenerateResponse, GenerateRequest, ClustersRequest, ClustersResponse
from app.utils.loader import load_and_validate_data, get_user_descriptions  # Keep for potential future use
from app.services.clusterer import create_clusterer
from app.services.llm_provider import create_llm_provider
from app.services.prompt_summarizer import create_prompt_summarizer
from app.services.rewriter import create_style_personalizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    # Startup
    logger.info("Starting up Personalized Query Rewriting service...")
    
    # Initialize services and store in app.state
    try:
        app.state.clusterer = create_clusterer(eps=0.3, min_samples=2)
        app.state.llm_provider = create_llm_provider()
        app.state.prompt_summarizer = create_prompt_summarizer(app.state.llm_provider)
        app.state.style_personalizer = create_style_personalizer(app.state.llm_provider)
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Personalized Query Rewriting service...")


# Dependency functions to get services from app.state
def get_clusterer(request: Request):
    """Get clusterer service from app state."""
    return request.app.state.clusterer


def get_llm_provider(request: Request):
    """Get LLM provider service from app state."""
    return request.app.state.llm_provider


def get_prompt_summarizer(request: Request):
    """Get prompt summarizer service from app state."""
    return request.app.state.prompt_summarizer


def get_style_personalizer(request: Request):
    """Get style personalizer service from app state."""
    return request.app.state.style_personalizer


# Create FastAPI app
app = FastAPI(
    title="Personalized Query Rewriting API",
    description="API for clustering and personalizing user queries using LLM",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Personalized Query Rewriting API is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Service is operational"}


@app.post("/generate", response_model=GenerateResponse)
async def generate_personalized_rewrites(
    request: GenerateRequest,
    clusterer=Depends(get_clusterer),
    prompt_summarizer=Depends(get_prompt_summarizer),
    style_personalizer=Depends(get_style_personalizer)
):
    """
    Generate clustered queries, rewritten representatives, and personalized rewrites.
    
    Args:
        request: GenerateRequest containing users and queries to process
        clusterer: QueryClusterer service from dependency injection
        prompt_summarizer: PromptSummarizer service from dependency injection
        style_personalizer: StylePersonalizer service from dependency injection
        
    Returns:
        GenerateResponse with clustered queries, representative rewrites, and personalized rewrites
    """
    try:
        # Extract data from request payload
        users = request.users
        queries = request.queries
        
        # Create user descriptions mapping
        user_descriptions = {user.user_id: user.description for user in users}
        
        logger.info(f"Processing {len(queries)} queries for {len(users)} users...")
        
        # Step 1: Cluster queries
        logger.info("Clustering queries...")
        clustered_queries = clusterer.process_queries(queries)
        logger.info(f"Created {len(clustered_queries)} clusters")
        
        # Step 2: Generate summarized prompts for clusters
        logger.info("Generating summarized prompts...")
        representative_rewrites = prompt_summarizer.summarize_clusters(clustered_queries)
        
        # Step 3: Create personalized rewrites
        logger.info("Creating personalized rewrites...")
        all_queries = []
        all_rewritten_queries = []
        
        for clustered_query, rewritten_query in zip(clustered_queries, representative_rewrites):
            # For each query in the cluster, use the rewritten representative
            for query in clustered_query.queries:
                all_queries.append(query)
                all_rewritten_queries.append(rewritten_query)
        
        personalized_rewrites = style_personalizer.create_personalized_rewrites(
            all_queries, all_rewritten_queries, user_descriptions
        )
        
        logger.info(f"Generated {len(personalized_rewrites)} personalized rewrites")
        
        # Create response
        response = GenerateResponse(
            clustered_queries=clustered_queries,
            representative_rewrites=representative_rewrites,
            personalized_rewrites=personalized_rewrites
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in generate endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/clusters", response_model=ClustersResponse)
async def get_clusters(
    request: ClustersRequest,
    clusterer=Depends(get_clusterer)
):
    """Get just the clustered queries without LLM processing."""
    try:
        # Extract data from request payload
        users = request.users
        queries = request.queries
        
        logger.info(f"Clustering {len(queries)} queries for {len(users)} users...")
        
        clustered_queries = clusterer.process_queries(queries)
        
        clusters_data = [
            {
                "cluster_id": cq.cluster_id,
                "user_ids": cq.user_ids,
                "query_count": len(cq.queries)
            }
            for cq in clustered_queries
        ]
        
        response = ClustersResponse(clusters=clusters_data)
        return response
        
    except Exception as e:
        logger.error(f"Error in clusters endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
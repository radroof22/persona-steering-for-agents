from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
import logging

from backend.models.schema import Query, GenerateResponse, GenerateRequest, ClustersRequest, ClustersResponse, PersonalizedQueryResponse
from backend.services.clusterer import QueryClusterer, create_clusterer
from backend.services.llm_provider import LLMProvider, create_llm_provider
from backend.services.prompt_summarizer import PromptSummarizer, create_prompt_summarizer
from backend.services.rewriter import StylePersonalizer, create_style_personalizer

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
        app.state.clusterer: QueryClusterer = create_clusterer(eps=0.3, min_samples=2)
        app.state.llm_provider: LLMProvider = create_llm_provider()
        app.state.prompt_summarizer: PromptSummarizer = create_prompt_summarizer(app.state.llm_provider)
        app.state.style_personalizer: StylePersonalizer = create_style_personalizer(app.state.llm_provider)
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Personalized Query Rewriting service...")


# Dependency functions to get services from app.state
def get_clusterer(request: Request) -> QueryClusterer:
    """Get clusterer service from app state."""
    return request.app.state.clusterer


def get_llm_provider(request: Request) -> LLMProvider:
    """Get LLM provider service from app state."""
    return request.app.state.llm_provider


def get_prompt_summarizer(request: Request) -> PromptSummarizer:
    """Get prompt summarizer service from app state."""
    return request.app.state.prompt_summarizer


def get_style_personalizer(request: Request) -> StylePersonalizer:
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
    style_personalizer=Depends(get_style_personalizer),
):
    """
    Orchestrate the full pipeline: clustering → summarization → personalization.
    
    Args:
        request: GenerateRequest containing users and queries to process
        clusterer: QueryClusterer service
        prompt_summarizer: PromptSummarizer service
        style_personalizer: StylePersonalizer service
        llm_provider: LLMProvider service
        
    Returns:
        GenerateResponse with list of PersonalizedQueryResponse objects
    """
    try:
        users = request.users
        queries = request.queries
        
        # Create user descriptions mapping
        user_descriptions = {user.user_id: user.description for user in users}
        
        logger.info(f"Processing {len(queries)} queries for {len(users)} users...")
        
        # Step 1: Cluster queries directly using the service
        logger.info("Step 1: Clustering queries...")
        clustered_queries = clusterer.process_queries(queries)
        logger.info(f"Created {len(clustered_queries)} clusters from {len(queries)} queries")
        
        # Step 2: For each cluster, summarize queries and answer summarized question using the service
        logger.info("Step 2: Summarizing clusters...")
        cluster_summaries = {}
        for cluster in clustered_queries:
            try:
                # Call the service method directly
                summarized_query = prompt_summarizer.summarize_cluster(cluster)
                summarized_response = f"Summary for cluster {cluster.cluster_id}: {summarized_query}"
                
                cluster_summaries[cluster.cluster_id] = {
                    "summarized_query": summarized_query,
                    "summarized_response": summarized_response,
                    "success": True,
                    "error_message": None
                }
            except Exception as e:
                logger.error(f"Failed to summarize cluster {cluster.cluster_id}: {e}")
                cluster_summaries[cluster.cluster_id] = {
                    "summarized_query": "",
                    "summarized_response": "",
                    "success": False,
                    "error_message": str(e)
                }
        
        # Step 3: For each original query, personalize using the service
        logger.info("Step 3: Personalizing responses...")
        results = []
        
        for cluster in clustered_queries:
            cluster_summary = cluster_summaries[cluster.cluster_id]

            if len(cluster.queries) == 1:
                results.append(
                    PersonalizedQueryResponse(
                        original_query=cluster.queries[0].question,
                        user_id=cluster.queries[0].user_id,
                        cluster_id=cluster.cluster_id,
                        summarized_query=cluster_summary["summarized_query"],
                        summarized_response=cluster_summary["summarized_response"],
                        personalized_response=cluster_summary["summarized_response"],
                        success=True,
                        error_message=None
                    )
                )
                continue
            
            for query in cluster.queries:
                try:
                    if cluster_summary["success"]:
                        # Get user description
                        user_description = user_descriptions.get(query.user_id, "")
                        
                        # Call the service method directly
                        personalized_response = style_personalizer.personalize_response(
                            original_query=query.question,
                            user_description=user_description,
                            summarized_query=cluster_summary["summarized_query"],
                            summarized_response=cluster_summary["summarized_response"]
                        )
                        
                        # Create successful result
                        result = PersonalizedQueryResponse(
                            original_query=query.question,
                            user_id=query.user_id,
                            cluster_id=cluster.cluster_id,
                            summarized_query=cluster_summary["summarized_query"],
                            summarized_response=cluster_summary["summarized_response"],
                            personalized_response=personalized_response,
                            success=True
                        )
                    else:
                        # Summarization failed
                        result = PersonalizedQueryResponse(
                            original_query=query.question,
                            user_id=query.user_id,
                            cluster_id=cluster.cluster_id,
                            summarized_query="",
                            summarized_response="",
                            personalized_response="",
                            success=False,
                            error_message=f"Summarization failed: {cluster_summary['error_message']}"
                        )
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing query '{query.question}': {e}")
                    # Add failed result
                    result = PersonalizedQueryResponse(
                        original_query=query.question,
                        user_id=query.user_id,
                        cluster_id=cluster.cluster_id,
                        summarized_query="",
                        summarized_response="",
                        personalized_response="",
                        success=False,
                        error_message=str(e)
                    )
                    results.append(result)
        
        logger.info(f"Generated {len(results)} personalized query responses")
        
        # Create response
        response = GenerateResponse(results=results)
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
        
        response = ClustersResponse(clusters=clustered_queries)
        return response
        
    except Exception as e:
        logger.error(f"Error in clusters endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
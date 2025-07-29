from fastapi import FastAPI, HTTPException
from models import (
    UserRegistration, QueryBatch, User, ClusteringParameters, ClusteringResponse,
    RewriteRequest, RewriteResponse, StyleAnalysisResponse, QueryAnswerRequest, QueryAnswerResponse,
    BatchUserQuestionsRequest, BatchUserQuestionsResponse, PersonalizedResponse
)
from database import ParquetDatabase
from clustering import QueryClusterer
from rewrite import ResponseRewriter
from query_answerer import QueryAnswerer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentosa API", description="LLM Personalization Platform")

# Initialize database, clusterer, and rewriter
db = ParquetDatabase()
clusterer = QueryClusterer()

# Initialize rewriter with model (this may take a moment to load)
try:
    rewriter = ResponseRewriter(db)
    logger.info("ResponseRewriter initialized successfully with Hugging Face model")
except Exception as e:
    logger.error(f"Failed to initialize ResponseRewriter with model: {e}")
    logger.info("Falling back to basic functionality")
    rewriter = None

# Initialize query answerer with T5 model
try:
    answerer = QueryAnswerer(db)
    logger.info("QueryAnswerer initialized successfully with T5 model")
except Exception as e:
    logger.error(f"Failed to initialize QueryAnswerer with T5 model: {e}")
    logger.info("Query answering functionality will not be available")
    answerer = None

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    logger.info("Initializing Sentosa API...")

@app.post("/register", response_model=User)
async def register_user(user_data: UserRegistration):
    """
    Register a new user with optional sample text corpus and natural language description
    """
    try:
        # Add user to database
        new_user = db.add_user(
            user_id=user_data.user_id,
            sample_text_corpus=user_data.sample_text_corpus,
            natural_language_description=user_data.natural_language_description
        )
        
        # Print user information for verification
        print("=== NEW USER REGISTERED ===")
        print(f"User ID: {new_user['user_id']}")
        print(f"Unique ID: {new_user['unique_id']}")
        print(f"Registration Date: {new_user['registration_date']}")
        if new_user['sample_text_corpus']:
            print(f"Sample Text Corpus: {new_user['sample_text_corpus'][:100]}...")
        if new_user['natural_language_description']:
            print(f"Natural Language Description: {new_user['natural_language_description']}")
        print("==========================")
        
        return User(**new_user)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/batch-queries", response_model=ClusteringResponse)
async def process_batch_queries(query_batch: QueryBatch):
    """
    Process a batch of queries and identify clusters using DBSCAN
    """
    if not query_batch.queries:
        raise HTTPException(status_code=400, detail="No queries provided")
    
    try:
        # Add new queries to database
        db.add_queries(query_batch.queries)
        
        # Perform clustering only on the current batch of queries
        clusters, clustering_info = clusterer.cluster_queries(query_batch.queries)
        
        # Print cluster information for verification
        clusterer.print_cluster_summary(clustering_info)
        
        return ClusteringResponse(**clustering_info)
        
    except Exception as e:
        logger.error(f"Error processing batch queries: {e}")
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")

@app.get("/users")
async def get_users():
    """Get all registered users"""
    try:
        return db.get_all_users()
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Missing required dependency: {e}. Please ensure the conda environment is activated: 'mamba activate sentosa'"
        )
    except Exception as e:
        logger.error(f"Error retrieving users: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/queries")
async def get_queries():
    """Get all stored queries"""
    try:
        return db.get_all_queries()
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Missing required dependency: {e}. Please ensure the conda environment is activated: 'mamba activate sentosa'"
        )
    except Exception as e:
        logger.error(f"Error retrieving queries: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/answers")
async def get_answers():
    """Get all stored answers"""
    try:
        return db.get_all_answers()
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Missing required dependency: {e}. Please ensure the conda environment is activated: 'mamba activate sentosa'"
        )
    except Exception as e:
        logger.error(f"Error retrieving answers: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/configure-clustering")
async def configure_clustering(params: ClusteringParameters):
    """
    Configure clustering parameters
    """
    try:
        global clusterer
        clusterer = QueryClusterer(
            eps=params.eps,
            min_samples=params.min_samples,
            max_features=params.max_features,
            ngram_range=tuple(params.ngram_range)
        )
        logger.info(f"Updated clustering parameters: {params.dict()}")
        return {"message": "Clustering parameters updated successfully", "parameters": params.dict()}
    except Exception as e:
        logger.error(f"Error configuring clustering: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/clustering-stats")
async def get_clustering_stats():
    """
    Get clustering statistics and recommendations based on stored queries
    """
    try:
        all_queries = db.get_query_texts()
        if not all_queries:
            return {"message": "No queries available for analysis"}
        
        # Use a sample of queries for analysis (limit to prevent memory issues)
        sample_size = min(100, len(all_queries))
        sample_queries = all_queries[:sample_size]
        
        # Analyze distance distribution
        distance_stats = clusterer.analyze_distance_distribution(sample_queries)
        
        # Get epsilon recommendations
        recommendations = clusterer.get_optimal_epsilon_recommendations(distance_stats)
        
        return {
            "distance_statistics": distance_stats,
            "epsilon_recommendations": recommendations,
            "current_parameters": {
                "eps": clusterer.eps,
                "min_samples": clusterer.min_samples,
                "max_features": clusterer.max_features,
                "ngram_range": clusterer.ngram_range
            },
            "analysis_sample_size": sample_size,
            "total_stored_queries": len(all_queries)
        }
    except Exception as e:
        logger.error(f"Error getting clustering stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/rewrite", response_model=RewriteResponse)
async def rewrite_response(request: RewriteRequest):
    """
    Rewrite a response to match the user's familiar writing style
    """
    try:
        if rewriter is None:
            raise HTTPException(
                status_code=503, 
                detail="Rewrite service not available. Model failed to load."
            )
        
        # Rewrite the response
        rewritten_response = rewriter.rewrite_response(
            original_response=request.original_response,
            user_id=request.user_id
        )
        
        # Determine if style was applied
        style_applied = rewritten_response != request.original_response
        
        # Print rewrite information for verification
        print("=== RESPONSE REWRITE ===")
        print(f"User ID: {request.user_id}")
        print(f"Style applied: {style_applied}")
        print(f"Original: {request.original_response[:100]}...")
        print(f"Rewritten: {rewritten_response[:100]}...")
        print("========================")
        
        return RewriteResponse(
            original_response=request.original_response,
            rewritten_response=rewritten_response,
            user_id=request.user_id,
            style_applied=style_applied
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rewriting response: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/style-analysis/{user_id}", response_model=StyleAnalysisResponse)
async def get_user_style_analysis(user_id: str):
    """
    Get detailed analysis of user's writing style
    """
    try:
        if rewriter is None:
            raise HTTPException(
                status_code=503, 
                detail="Style analysis service not available. Model failed to load."
            )
        
        analysis = rewriter.get_user_style_analysis(user_id)
        
        if "error" in analysis:
            raise HTTPException(status_code=404, detail=analysis["error"])
        
        return StyleAnalysisResponse(**analysis)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing user style: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/answer-query", response_model=QueryAnswerResponse)
async def answer_query(request: QueryAnswerRequest):
    """
    Answer a query using T5 model with optional user personalization
    """
    try:
        if answerer is None:
            raise HTTPException(
                status_code=503, 
                detail="Query answering service not available. T5 model failed to load."
            )
        
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Answer the query using T5
        answer = answerer.answer_query(
            query=request.query,
            user_id=request.user_id
        )
        
        # Store the query-answer pair in database
        db.add_answer(
            query=request.query,
            answer=answer,
            user_id=request.user_id
        )
        
        # Determine if personalization was applied
        style_applied = request.user_id is not None
        
        # Print answer information for verification
        print("=== QUERY ANSWERED ===")
        print(f"Query: {request.query}")
        print(f"User ID: {request.user_id}")
        print(f"Style applied: {style_applied}")
        print(f"Answer: {answer[:200]}...")
        print("======================")
        
        return QueryAnswerResponse(
            query=request.query,
            answer=answer,
            user_id=request.user_id,
            style_applied=style_applied
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error answering query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/batch-user-questions", response_model=BatchUserQuestionsResponse)
async def process_batch_user_questions(request: BatchUserQuestionsRequest):
    """
    Process a batch of user-question pairs through the full pipeline:
    1. Extract questions for clustering
    2. Perform clustering on questions
    3. Answer each question with personalization
    4. Return personalized responses for all questions
    """
    if not request.user_questions:
        raise HTTPException(status_code=400, detail="No user questions provided")
    
    try:
        # Extract all questions for clustering
        questions = [uq.question for uq in request.user_questions]
        
        # Step 1: Add questions to database
        db.add_queries(questions)
        
        # Step 2: Perform clustering on all questions
        clusters, clustering_info = clusterer.cluster_queries(questions)
        
        # Create a mapping from question to cluster_id for later use
        question_to_cluster = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id != -1:  # Not noise
                question_to_cluster[questions[i]] = f"cluster_{cluster_id}"
            else:
                question_to_cluster[questions[i]] = "noise"
        
        # Step 3: Process each user-question pair with personalization
        responses = []
        
        for user_question in request.user_questions:
            try:
                # Answer the question with personalization
                answer = answerer.answer_query(
                    query=user_question.question,
                    user_id=user_question.user_id
                )
                
                # Store the query-answer pair in database
                db.add_answer(
                    query=user_question.question,
                    answer=answer,
                    user_id=user_question.user_id
                )
                
                # Create personalized response
                personalized_response = PersonalizedResponse(
                    user_id=user_question.user_id,
                    question=user_question.question,
                    answer=answer,
                    style_applied=True,  # Always True since we have user_id
                    cluster_id=question_to_cluster.get(user_question.question)
                )
                
                responses.append(personalized_response)
                
            except Exception as e:
                logger.error(f"Error processing question for user {user_question.user_id}: {e}")
                # Create error response
                error_response = PersonalizedResponse(
                    user_id=user_question.user_id,
                    question=user_question.question,
                    answer=f"Error processing question: {str(e)}",
                    style_applied=False,
                    cluster_id=question_to_cluster.get(user_question.question)
                )
                responses.append(error_response)
        
        # Print batch processing information for verification
        print("=== BATCH USER QUESTIONS PROCESSED ===")
        print(f"Total questions: {len(request.user_questions)}")
        print(f"Clusters found: {clustering_info['n_clusters']}")
        print(f"Noise points: {clustering_info['n_noise']}")
        print("Responses:")
        for response in responses:
            print(f"  User: {response.user_id}")
            print(f"  Question: {response.question}")
            print(f"  Cluster: {response.cluster_id}")
            print(f"  Answer: {response.answer[:100]}...")
            print()
        print("======================================")
        
        return BatchUserQuestionsResponse(
            total_questions=len(request.user_questions),
            responses=responses,
            clustering_info=clustering_info
        )
        
    except Exception as e:
        logger.error(f"Error processing batch user questions: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
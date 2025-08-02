from pydantic import BaseModel, Field
from typing import List, Optional


class User(BaseModel):
    """User model with user_id and description."""
    user_id: str = Field(..., description="Unique user identifier")
    description: str = Field(..., description="User description for personalization")


class Query(BaseModel):
    """Query model with user_id and question (incoming from customers)."""
    user_id: str = Field(..., description="User who made the query")
    question: str = Field(..., description="The user's question/query")


class ProcessedQuery(BaseModel):
    """Query model with cluster information (internal processing)."""
    user_id: str = Field(..., description="User who made the query")
    question: str = Field(..., description="The user's question/query")
    cluster_id: int = Field(..., description="Cluster identifier for this query")


class ClusteredQuery(BaseModel):
    """Represents a cluster of similar queries."""
    cluster_id: int = Field(..., description="Unique cluster identifier")
    user_ids: List[str] = Field(..., description="List of user IDs in this cluster")
    queries: List[Query] = Field(..., description="All queries in this cluster")


class PersonalizedQueryResponse(BaseModel):
    """Final response for a single query with all processing results."""
    original_query: str = Field(..., description="Original user query")
    user_id: str = Field(..., description="User identifier")
    cluster_id: int = Field(..., description="Cluster this query belongs to")
    summarized_query: str = Field(..., description="Summarized query for the cluster")
    summarized_response: str = Field(..., description="LLM response to summarized query")
    personalized_response: str = Field(..., description="Personalized response for the user")
    success: bool = Field(..., description="Whether processing was successful")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")


# Legacy model for backward compatibility
class PersonalizedRewrite(BaseModel):
    """Personalized rewrite for a specific user."""
    user_id: str = Field(..., description="User identifier")
    original_query: str = Field(..., description="Original user query")
    rewritten_query: str = Field(..., description="LLM rewritten query")
    personalized_query: str = Field(..., description="Style-personalized query")


class GenerateRequest(BaseModel):
    """Request model for the /generate endpoint."""
    users: List[User] = Field(..., description="List of users with their descriptions")
    queries: List[Query] = Field(..., description="List of queries to process")


class ClustersRequest(BaseModel):
    """Request model for the /clusters endpoint."""
    users: List[User] = Field(..., description="List of users with their descriptions")
    queries: List[Query] = Field(..., description="List of queries to cluster")


# New microservice endpoints
class SummarizeRequest(BaseModel):
    """Request model for the /summarize endpoint."""
    cluster: ClusteredQuery = Field(..., description="Single cluster to summarize")


class SummarizeResponse(BaseModel):
    """Response model for the /summarize endpoint."""
    cluster_id: int = Field(..., description="Cluster identifier")
    summarized_query: str = Field(..., description="Summarized query for the cluster")
    summarized_response: str = Field(..., description="LLM response to summarized query")
    success: bool = Field(..., description="Whether summarization was successful")
    error_message: Optional[str] = Field(None, description="Error message if summarization failed")


class PersonalizeRequest(BaseModel):
    """Request model for the /personalize endpoint."""
    original_query: str = Field(..., description="Original user query")
    user_description: str = Field(..., description="User description for personalization")
    summarized_query: str = Field(..., description="Summarized query for context")
    summarized_response: str = Field(..., description="Summarized response to personalize")


class PersonalizeResponse(BaseModel):
    """Response model for the /personalize endpoint."""
    personalized_response: str = Field(..., description="Personalized response for the user")
    success: bool = Field(..., description="Whether personalization was successful")
    error_message: Optional[str] = Field(None, description="Error message if personalization failed")


# Updated generate response
class GenerateResponse(BaseModel):
    """Response model for the /generate endpoint."""
    results: List[PersonalizedQueryResponse] = Field(..., description="List of personalized query responses")




class ClustersResponse(BaseModel):
    """Response model for the /clusters endpoint."""
    clusters: List[ClusteredQuery] = Field(..., description="List of clustered queries") 
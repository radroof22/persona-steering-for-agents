from pydantic import BaseModel, Field
from typing import List, Optional


class User(BaseModel):
    """User model with user_id and description."""
    user_id: str = Field(..., description="Unique user identifier")
    description: str = Field(..., description="User description for personalization")


class Query(BaseModel):
    """Query model with user_id and question."""
    user_id: str = Field(..., description="User who made the query")
    question: str = Field(..., description="The user's question/query")


class ClusteredQuery(BaseModel):
    """Represents a cluster of similar queries."""
    cluster_id: int = Field(..., description="Unique cluster identifier")
    user_ids: List[str] = Field(..., description="List of user IDs in this cluster")
    queries: List[Query] = Field(..., description="All queries in this cluster")


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


class GenerateResponse(BaseModel):
    """Response model for the /generate endpoint."""
    clustered_queries: List[ClusteredQuery] = Field(..., description="Clustered queries")
    representative_rewrites: List[str] = Field(..., description="LLM rewritten representatives")
    personalized_rewrites: List[PersonalizedRewrite] = Field(..., description="Personalized rewrites per user")


class ClustersResponse(BaseModel):
    """Response model for the /clusters endpoint."""
    clusters: List[dict] = Field(..., description="List of clusters with their metadata") 
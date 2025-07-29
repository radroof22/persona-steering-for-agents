from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class UserRegistration(BaseModel):
    """Model for user registration requests"""
    user_id: str
    sample_text_corpus: Optional[str] = None
    natural_language_description: Optional[str] = None

class QueryBatch(BaseModel):
    """Model for batch query processing requests"""
    queries: List[str]

class User(BaseModel):
    """Model for user responses"""
    user_id: str
    sample_text_corpus: Optional[str] = None
    natural_language_description: Optional[str] = None
    registration_date: str
    unique_id: str

class ClusteringParameters(BaseModel):
    """Model for clustering parameter configuration"""
    eps: float = 1.2
    min_samples: int = 2
    max_features: int = 1000
    ngram_range: List[int] = [1, 2]

class ClusteringResponse(BaseModel):
    """Model for clustering response"""
    total_queries: int
    clusters: List[dict]
    clustering_algorithm: str
    parameters: dict
    n_clusters: int
    n_noise: int

class RewriteRequest(BaseModel):
    """Model for rewrite requests"""
    original_response: str
    user_id: str

class RewriteResponse(BaseModel):
    """Model for rewrite responses"""
    original_response: str
    rewritten_response: str
    user_id: str
    style_applied: bool

class StyleAnalysisResponse(BaseModel):
    """Model for style analysis responses"""
    user_id: str
    style_characteristics: dict
    interpretation: dict
    sample_text_length: int

class QueryAnswerRequest(BaseModel):
    """Model for query answering requests"""
    query: str
    user_id: Optional[str] = None

class QueryAnswerResponse(BaseModel):
    """Model for query answering responses"""
    query: str
    answer: str
    user_id: Optional[str] = None
    style_applied: bool

class UserQuestionPair(BaseModel):
    """Model for a single user-question pair"""
    user_id: str
    question: str

class BatchUserQuestionsRequest(BaseModel):
    """Model for batch user questions processing requests"""
    user_questions: List[UserQuestionPair]

class PersonalizedResponse(BaseModel):
    """Model for a personalized response to a user question"""
    user_id: str
    question: str
    answer: str
    style_applied: bool
    cluster_id: Optional[str] = None

class BatchUserQuestionsResponse(BaseModel):
    """Model for batch user questions processing responses"""
    total_questions: int
    responses: List[PersonalizedResponse]
    clustering_info: Optional[Dict[str, Any]] = None 
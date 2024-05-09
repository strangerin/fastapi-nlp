from pydantic import BaseModel


class SimilarityRequest(BaseModel):
    method: str
    line1: str
    line2: str


class SimilarityResponse(SimilarityRequest):
    similarity: float


class ReviewClassificationRequest(BaseModel):
    review_text: str


class ReviewClassificationResponse(BaseModel):
    review_text: str
    sentiment: int

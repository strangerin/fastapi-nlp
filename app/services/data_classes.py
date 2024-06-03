from pydantic import BaseModel
from typing import List


class SimilarityRequest(BaseModel):
    method: str
    line1: str
    line2: str


class SimilarityResponse(SimilarityRequest):
    similarity: float


class ReviewClassificationRequest(BaseModel):
    review_text: str
    preprocess_method: str


class ReviewClassificationResponse(BaseModel):
    review_text: str
    sentiment: int


class GroupSentencesRequest(BaseModel):
    sentences: List[str]


class GroupSentencesResponse(BaseModel):
    groups: List[List[str]]

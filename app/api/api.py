from typing import Any

from fastapi import APIRouter, Request
from textdistance import hamming, levenshtein, jaro_winkler, jaccard, sorensen_dice

from app.models.predict import PredictRequest, PredictResponse
from app.services.data_classes import SimilarityRequest, SimilarityResponse

api_router = APIRouter()


@api_router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, payload: PredictRequest) -> Any:
    """
    ML Prediction API
    """
    input_text = payload.input_text
    model = request.app.state.model

    predict_value = model.predict(input_text)
    return PredictResponse(result=predict_value)


@api_router.post("/similarity", response_model=SimilarityResponse)
async def calculate_similarity(request: SimilarityRequest) -> SimilarityResponse:
    """
    Similarity Calculation API
    """
    method = request.method
    line1 = request.line1
    line2 = request.line2
    # TODO add /help endpoint
    if method == "hamming":
        similarity = hamming.normalized_similarity(line1, line2)
    elif method == "levenshtein":
        similarity = levenshtein.normalized_similarity(line1, line2)
    elif method == "jaro_winkler":
        similarity = jaro_winkler.normalized_similarity(line1, line2)
    elif method == "jaccard":
        similarity = jaccard.normalized_similarity(line1, line2)
    elif method == "sorensen_dice":
        similarity = sorensen_dice.normalized_similarity(line1, line2)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return SimilarityResponse(
        method=method,
        line1=line1,
        line2=line2,
        similarity=similarity
    )


@api_router.get("/help")
async def help_endpoint():
    """
    Help Endpoint
    """
    help_text = """
    Available Endpoints:

    1. /predict (POST):
       - Description: ML Prediction API
       - Request Body:
         - input_text: str (required)
       - Response:
         - result: str

    2. /similarity (POST):
       - Description: Similarity Calculation API
       - Request Body:
         - method: str (required, options: hamming, levenshtein, jaro_winkler, jaccard, sorensen_dice)
         - line1: str (required)
         - line2: str (required)
       - Response:
         - method: str
         - line1: str
         - line2: str
         - similarity: float

    3. /help (GET):
       - Description: Help Endpoint
       - Response: This help text
    """
    return help_text

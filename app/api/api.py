from typing import Any

from fastapi import APIRouter, Request
from textdistance import hamming, levenshtein, jaro_winkler, jaccard, sorensen_dice

from app.models.predict import PredictRequest, PredictResponse
from app.services.data_classes import SimilarityRequest, SimilarityResponse, ReviewClassificationRequest, \
    ReviewClassificationResponse
from app.models.model_loader import load_model_and_vectorizer
from app.services.utils import preprocess_text


api_router = APIRouter()

# Load the trained model and vectorizer
model, vectorizer = load_model_and_vectorizer()


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


@api_router.post("/classify_review", response_model=ReviewClassificationResponse)
async def classify_review(request: ReviewClassificationRequest) -> ReviewClassificationResponse:
    """
    Review Classification API
    """
    review_text = request.review_text

    # Preprocess the review text
    preprocessed_text = preprocess_text(review_text)

    # Convert text to numerical features using the loaded vectorizer
    text_features = vectorizer.transform([preprocessed_text])

    # Make predictions using the loaded model
    prediction = model.predict(text_features)[0]

    return ReviewClassificationResponse(review_text=review_text, sentiment=prediction)


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

    3. /classify_review (POST):
       - Description: Review Classification API
       - Request Body:
         - review_text: str (required)
       - Response:
         - review_text: str
         - sentiment: int (0 for negative, 1 for positive)

    4. /help (GET):
       - Description: Help Endpoint
       - Response: This help text
    """
    return help_text

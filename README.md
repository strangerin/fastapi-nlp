# FastAPI NLP server

TODO:
Complete description, add model cards, etc

## Available Endpoints

- `/api/v1/predict`: ML Prediction API
- `/api/v1/similarity`: Similarity Calculation API
- `/api/v1/classify_review`: Review Classification API
- `/api/v1/group_sentences`: Sentence Grouping API
- `/api/v1/analyze_sentiment`: Sentiment Analysis API using a transformer model

### Sentiment Analysis API

```sh
$ curl --request POST --url http://127.0.0.1:9000/api/v1/analyze_sentiment --header 'Content-Type: application/json' --data '{"text": "I love this product!"}'


## Run Web API
### Local

```sh
$ sh run.sh
```

```sh
$ poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 9000
```

### Docker
```sh
$ docker build -f Dockerfile -t fastapi-ml .
$ docker run -p 9000:9000 --rm --name fastapi-ml -t -i fastapi-ml
```

### Docker Compose

```sh
$ docker compose up --build
```

## Request Commands

```sh 
$ curl --request POST --url http://127.0.0.1:9000/api/v1/predict --header 'Content-Type: application/json' --data '{"input_text": "test"}'
```

```sh
$ http POST http://127.0.0.1:9000/api/v1/predict input_text=テスト
```

## Development
### Run Tests and Linter

```
$ poetry run tox
```

## Reference

- [tiangolo/full\-stack\-fastapi\-postgresql: Full stack, modern web application generator\. Using FastAPI, PostgreSQL as database, Docker, automatic HTTPS and more\.](https://github.com/tiangolo/full-stack-fastapi-postgresql)
- [eightBEC/fastapi\-ml\-skeleton: FastAPI Skeleton App to serve machine learning models production\-ready\.](https://github.com/eightBEC/fastapi-ml-skeleton)

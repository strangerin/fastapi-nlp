import nltk
import re
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag
from nltk.corpus import wordnet
from string import punctuation


def _download_resources():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')


def load_spacy_model(model_name="en_core_web_sm"):
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"Downloading spaCy model: {model_name}")
        spacy.cli.download(model_name)
        nlp = spacy.load(model_name)
    return nlp


def preprocess_text(text):
    # TODO move to a Dockerfile, it is stupid ad-hoc solution
    # download once, cache later
    _download_resources()
    # Tokenization
    tokens = word_tokenize(text)

    # Lowercase conversion
    tokens = [token.lower() for token in tokens]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Remove non-alphabetic tokens
    tokens = [token for token in tokens if token.isalpha()]

    # Join tokens back into a string
    processed_text = ' '.join(tokens)

    return processed_text

# TODO opt out for a simpler pipeline, this one is excessive, used model was trained with the previous one
# def preprocess_text(texts, pipeline=None, lowercase=True, remove_stopwords=True, stemming=False, lemmatization=True,
#                     remove_punctuation=True, remove_numbers=True, remove_short_words=True, min_word_length=2,
#                     remove_custom_words=None, keep_alpha_only=False, pos_tagging=False, remove_repeated_chars=False,
#                     max_repeated_chars=2, replace_urls=True, replace_emails=True, replace_phone_numbers=True,
#                     replace_currency_symbols=True, replace_custom_patterns=None):
#     processed_texts = []
#     for text in texts:
#         # Tokenization
#         sentences = sent_tokenize(text)
#         tokens = []
#         for sentence in sentences:
#             if remove_punctuation:
#                 tokenizer = RegexpTokenizer(r'\w+')
#                 sentence_tokens = tokenizer.tokenize(sentence)
#             else:
#                 sentence_tokens = word_tokenize(sentence)
#             tokens.extend(sentence_tokens)
#
#         # Lowercase conversion
#         if lowercase:
#             tokens = [token.lower() for token in tokens]
#
#         # Remove stopwords
#         if remove_stopwords:
#             stop_words = set(stopwords.words('english'))
#             tokens = [token for token in tokens if token not in stop_words]
#
#         # Remove custom words
#         if remove_custom_words:
#             tokens = [token for token in tokens if token not in remove_custom_words]
#
#         # Keep only alphabetic characters
#         if keep_alpha_only:
#             tokens = [token for token in tokens if token.isalpha()]
#
#         # Remove numbers
#         if remove_numbers:
#             tokens = [token for token in tokens if not token.isnumeric()]
#
#         # Remove short words
#         if remove_short_words:
#             tokens = [token for token in tokens if len(token) >= min_word_length]
#
#         # Stemming
#         if stemming:
#             stemmer = PorterStemmer()
#             tokens = [stemmer.stem(token) for token in tokens]
#
#         # Lemmatization
#         if lemmatization:
#             lemmatizer = WordNetLemmatizer()
#             tokens = [lemmatizer.lemmatize(token) for token in tokens]
#
#         # POS tagging
#         if pos_tagging:
#             tokens = pos_tag(tokens)
#
#         # Remove repeated characters
#         if remove_repeated_chars:
#             tokens = [re.sub(r'(.)\1+', r'\1' * max_repeated_chars, token) for token in tokens]
#
#         # Replace URLs
#         if replace_urls:
#             url_pattern = re.compile(r'https?://\S+|www\.\S+')
#             tokens = [re.sub(url_pattern, 'URL', token) for token in tokens]
#
#         # Replace email addresses
#         if replace_emails:
#             email_pattern = re.compile(r'\S+@\S+')
#             tokens = [re.sub(email_pattern, 'EMAIL', token) for token in tokens]
#
#         # Replace phone numbers
#         if replace_phone_numbers:
#             phone_pattern = re.compile(r'\d{3}[-.\s]??\d{3}[-.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-.\s]??\d{4}|\d{3}[-.\s]??\d{4}')
#             tokens = [re.sub(phone_pattern, 'PHONE', token) for token in tokens]
#
#         # Replace currency symbols
#         if replace_currency_symbols:
#             currency_pattern = re.compile(r'[\$€£¥₹₽]')
#             tokens = [re.sub(currency_pattern, 'CURRENCY', token) for token in tokens]
#
#         # Replace custom patterns
#         if replace_custom_patterns:
#             for pattern, replacement in replace_custom_patterns.items():
#                 tokens = [re.sub(pattern, replacement, token) for token in tokens]
#
#         # Custom pipeline
#         if pipeline is not None:
#             for func in pipeline:
#                 tokens = func(tokens)
#
#         # Join tokens back into a string
#         processed_text = ' '.join(tokens)
#         processed_texts.append(processed_text)
#
#     return processed_texts

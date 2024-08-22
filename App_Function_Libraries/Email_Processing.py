# email_processing.py

import email
from email.utils import parsedate_to_datetime, getaddresses
import re
from typing import Dict, Any, List
from bs4 import BeautifulSoup
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
from textblob import TextBlob

nlp = spacy.load("en_core_web_sm")

#contains all the functions for parsing emails, extracting metadata, calculating relevance scores, performing entity recognition, topic clustering, sentiment analysis, and keyword extraction.


def parse_email(email_content: str) -> Dict[str, Any]:
    msg = email.message_from_string(email_content)

    body = ""
    html_body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body = part.get_payload(decode=True).decode()
            elif part.get_content_type() == "text/html":
                html_body = part.get_payload(decode=True).decode()
    else:
        body = msg.get_payload(decode=True).decode()

    if not body and html_body:
        body = BeautifulSoup(html_body, "html.parser").get_text()

    return {
        'subject': msg['subject'],
        'from': msg['from'],
        'to': msg['to'],
        'cc': msg['cc'],
        'date': parsedate_to_datetime(msg['date']),
        'body': body,
        'message_id': msg['message-id'],
        'in_reply_to': msg['in-reply-to'],
        'references': msg['references']
    }


def extract_metadata(parsed_email: Dict[str, Any]) -> Dict[str, Any]:
    metadata = {
        'subject': parsed_email['subject'],
        'from': getaddresses([parsed_email['from']]),
        'to': getaddresses([parsed_email['to']]),
        'cc': getaddresses([parsed_email['cc']]) if parsed_email['cc'] else [],
        'date': parsed_email['date'],
        'email_addresses': re.findall(r'[\w\.-]+@[\w\.-]+', parsed_email['body']),
        'urls': re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                           parsed_email['body']),
        'message_id': parsed_email['message_id'],
        'in_reply_to': parsed_email['in_reply_to'],
        'references': parsed_email['references']
    }
    return metadata


def calculate_relevance_score(email_body: str, query: str) -> float:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([email_body, query])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return float(cosine_sim[0][0])


def extract_entities(text: str) -> Dict[str, list]:
    doc = nlp(text)
    entities = {
        'PERSON': [],
        'ORG': [],
        'GPE': [],  # Geopolitical Entity
        'DATE': [],
        'MONEY': []
    }
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    return entities


def perform_topic_clustering(email_bodies: List[str], num_topics: int = 5) -> List[Dict[str, Any]]:
    def preprocess(text):
        return [token for token in simple_preprocess(text) if token not in STOPWORDS]

    processed_docs = [preprocess(doc) for doc in email_bodies]
    dictionary = corpora.Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100,
                         update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)

    topics = lda_model.print_topics(num_words=10)
    return [{'id': topic[0], 'words': dict(word.split('*') for word in topic[1].split(' + '))} for topic in topics]


def analyze_email_thread(emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    thread_map = {}
    for email in emails:
        message_id = email['metadata']['message_id']
        in_reply_to = email['metadata']['in_reply_to']
        if in_reply_to:
            if in_reply_to in thread_map:
                thread_map[in_reply_to]['replies'].append(email)
            else:
                thread_map[in_reply_to] = {'email': None, 'replies': [email]}
        if message_id:
            if message_id in thread_map:
                thread_map[message_id]['email'] = email
            else:
                thread_map[message_id] = {'email': email, 'replies': []}

    threads = []
    for message_id, thread_info in thread_map.items():
        if not thread_info['email']['metadata']['in_reply_to']:
            threads.append(thread_info)

    return threads


def perform_sentiment_analysis(text: str) -> Dict[str, float]:
    blob = TextBlob(text)
    return {
        'polarity': blob.sentiment.polarity,
        'subjectivity': blob.sentiment.subjectivity
    }


def extract_keywords(text: str, num_keywords: int = 10) -> List[str]:
    doc = nlp(text)
    keywords = []
    for token in doc:
        if not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'PROPN', 'ADJ']:
            keywords.append(token.text.lower())
    return list(set(keywords))[:num_keywords]
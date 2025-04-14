from openai import OpenAI
import base64
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from uuid import uuid4

client = None

def load_openai(api_key):
    global client
    client = OpenAI(api_key=api_key)

def get_image_description(image_bytes):
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                }}
            ]}
        ]
    )
    return response.choices[0].message.content.strip()

def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding

def connect_qdrant(url, api_key=None):
    return QdrantClient(url=url, api_key=api_key)

def ensure_collection(client, collection_name):
    if not client.collection_exists(collection_name=collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=3072,  # embedding size for text-embedding-3-large
                distance=Distance.COSINE
            )
        )

def add_image_to_qdrant(client, collection, image_id, embedding, payload):
    point = PointStruct(
        id=str(image_id),  # Make sure ID is a string
        vector=embedding,
        payload=payload
    )
    client.upsert(collection_name=collection, points=[point])

def search_qdrant(client, collection, embedding, limit=5):
    return client.search(
        collection_name=collection,
        query_vector=embedding,
        limit=limit
    )

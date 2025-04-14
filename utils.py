from openai import OpenAI
import base64
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def get_embedding(text: str) -> list[float]:
    res = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return res.data[0].embedding

def describe_image(image_b64: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in detail."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                ]
            }
        ]
    )
    return response.choices[0].message.content

def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=[text]
    )
    return response.data[0].embedding

def connect_qdrant(url, api_key=None):
    return QdrantClient(url=url, api_key=api_key)

def ensure_collection(client, collection_name):
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
        )

def add_image_to_qdrant(client, collection, image_id, embedding, payload):
    point = PointStruct(
        id=image_id,
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

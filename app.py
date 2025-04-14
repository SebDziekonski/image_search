import streamlit as st
import os
import uuid
from dotenv import load_dotenv
from PIL import Image
import io

from utils import (
    load_openai,
    get_image_description,
    embed_text,
    connect_qdrant,
    ensure_collection,
    add_image_to_qdrant,
    search_qdrant
)

# Load env variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api = os.getenv("QDRANT_API_KEY")

# Constants
COLLECTION_NAME = "image_descriptions"

# Setup
load_openai(openai_key)
qdrant_client = connect_qdrant(qdrant_url, qdrant_api)
ensure_collection(qdrant_client, COLLECTION_NAME)

st.title("🖼️ Image Semantic Search with GPT-4o + Qdrant")

tab1, tab2 = st.tabs(["Upload Images", "Search by Description"])

with tab1:
    st.header("Upload and Process Images")
    uploaded_files = st.file_uploader("Upload image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.image(uploaded_file, caption=uploaded_file.name)
            bytes_data = uploaded_file.read()

            # Get description
            description = get_image_description(bytes_data)
            st.write(f"**Description:** {description}")

            # Embed
            embedding = embed_text(description)

            # Save to Qdrant
            image_id = (str(uuid.uuid4()))
            add_image_to_qdrant(
                qdrant_client,
                COLLECTION_NAME,
                image_id,
                embedding,
                payload={"filename": uploaded_file.name, "description": description}
            )
        st.success("Images processed and saved!")

with tab2:
    st.header("Search Images by Description")
    query = st.text_input("Enter your image description:")

    if query:
        query_embedding = embed_text(query)
        results = search_qdrant(qdrant_client, COLLECTION_NAME, query_embedding)

        st.subheader("Results")
        for result in results:
            st.write(f"**Filename**: {result.payload.get('filename')}")
            st.write(f"**Description**: {result.payload.get('description')}")

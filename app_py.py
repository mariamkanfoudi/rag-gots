import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from groq import Groq

COLLECTION_NAME = "gots_standard"
LLM_MODEL = "llama-3.3-70b-versatile"

@st.cache_resource
def load_clients():
    model = SentenceTransformer("BAAI/bge-m3")
    qdrant = QdrantClient(
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"]
    )
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    return model, qdrant, groq_client

model, qdrant_client, groq_client = load_clients()

st.title("RAG GOTS v7.0 — Llama 3.3 70B + Qdrant")

question = st.text_input("Votre question sur la norme GOTS :")

if question:
    with st.spinner("Recherche en cours..."):
        query_vector = model.encode(question, normalize_embeddings=True).tolist()
        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=5,
            with_payload=True
        )
        context = "\n\n---\n\n".join(
            [f"[Page {r.payload['page']}]\n{r.payload['text']}" for r in results.points]
        )
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "Tu es un expert GOTS. Réponds en français depuis le contexte."},
                {"role": "user", "content": f"Contexte:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0.2,
            max_tokens=1024
        )
        st.markdown(response.choices[0].message.content)
        with st.expander("Sources"):
            for r in results.points:
                st.write(f"Page {r.payload['page']} — score: {r.score:.3f}")

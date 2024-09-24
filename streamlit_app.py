import streamlit as st
import openai
import requests
import os
from authlib.integrations.requests_client import OAuth2Session
import faiss
import numpy as np

# Configuration des variables via les secrets Streamlit
DISCORD_CLIENT_ID = st.secrets["DISCORD_CLIENT_ID"]
DISCORD_CLIENT_SECRET = st.secrets["DISCORD_CLIENT_SECRET"]
DISCORD_REDIRECT_URI = st.secrets["DISCORD_REDIRECT_URI"]
DISCORD_API_ENDPOINT = "https://discord.com/api/v10"
DISCORD_GUILD_ID = st.secrets["DISCORD_GUILD_ID"]
DISCORD_REQUIRED_ROLE = st.secrets["DISCORD_REQUIRED_ROLE"]
OPENAI_API_KEY = st.secrets["openai_key"]

# Charger les fichiers du dossier data
def load_dataset():
    texts = []
    for file_name in os.listdir("./data"):
        if file_name.endswith(".txt"):
            with open(os.path.join("./data", file_name), "r", encoding="utf-8") as file:
                texts.append(file.read())
    return texts

# Utiliser OpenAI pour créer des embeddings
def get_embeddings(texts):
    openai.api_key = OPENAI_API_KEY
    embeddings = []
    for text in texts:
        response = openai.Embedding.create(input=text, engine="text-embedding-ada-002")
        embeddings.append(response['data'][0]['embedding'])
    return np.array(embeddings)

# Initialisation Faiss avec des embeddings
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # Taille de l'embedding (généralement 1536 pour OpenAI)
    index = faiss.IndexFlatL2(dimension)  # L2 distance
    index.add(embeddings)
    return index

# Requête Faiss
def query_faiss(index, query_embedding, texts):
    D, I = index.search(query_embedding, k=1)  # Trouver la correspondance la plus proche
    return texts[I[0][0]]

# Fonction pour l'authentification Discord OAuth2
def get_token(auth_code):
    token_url = "https://discord.com/api/oauth2/token"
    data = {
        'client_id': DISCORD_CLIENT_ID,
        'client_secret': DISCORD_CLIENT_SECRET,
        'grant_type': 'authorization_code',
        'code': auth_code,
        'redirect_uri': DISCORD_REDIRECT_URI,
    }
    response = requests.post(token_url, data=data)
    return response.json()

def get_user_roles(token):
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{DISCORD_API_ENDPOINT}/users/@me/guilds/{DISCORD_GUILD_ID}/member"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()["roles"]
    else:
        return []

def check_user_role(roles, required_role):
    return required_role in roles

# Authentification Discord
auth_code = st.query_params.get('code')

if auth_code:
    token = get_token(auth_code[0])
    user_roles = get_user_roles(token['access_token'])

    if check_user_role(user_roles, DISCORD_REQUIRED_ROLE):
        st.success("Accès autorisé au chatbot")

        # Charger et traiter les données
        texts = load_dataset()
        embeddings = get_embeddings(texts)
        index = create_faiss_index(embeddings)

        # Interface utilisateur
        query = st.text_input("Pose-moi une question :")
        if query:
            query_embedding = get_embeddings([query])  # Embedding de la requête
            result = query_faiss(index, query_embedding, texts)
            st.write(f"Réponse basée sur le dataset : {result}")

    else:
        st.error("Accès refusé. Vous n'avez pas le rôle requis.")
else:
    auth_url = f"https://discord.com/oauth2/authorize?client_id={DISCORD_CLIENT_ID}&redirect_uri={DISCORD_REDIRECT_URI}&response_type=code&scope=identify%20guilds"
    st.write(f"[Connecte-toi avec Discord]({auth_url})")

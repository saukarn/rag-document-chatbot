import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

print("PINECONE KEY LOADED:", bool(api_key))

pc = Pinecone(api_key=api_key)

print(pc.list_indexes())
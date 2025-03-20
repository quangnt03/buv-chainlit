import os
from sqlalchemy import make_url
from dotenv import load_dotenv, find_dotenv
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
import psycopg2

load_dotenv(find_dotenv())

connection_string = os.getenv('POSTGRES_URL')
db_name = os.getenv('POSTGRES_DB')
conn = psycopg2.connect(connection_string)
conn.autocommit = True

with conn.cursor() as c:
    c.execute(f"DROP DATABASE IF EXISTS {db_name}")
    c.execute(f"CREATE DATABASE {db_name}")
    
url = make_url(connection_string)
vector_store = PGVectorStore.from_params(
    database=db_name,
    host=url.host,
    password=url.password,
    port=url.port,
    user=url.username,
    table_name=os.getenv('POSTGRES_DB_TABLE'),
    hybrid_search=True,
    embed_dim=1536,  # openai embedding dimension
    hnsw_kwargs={
        "hnsw_m": 16,
        "hnsw_ef_construction": 64,
        "hnsw_ef_search": 300,
        "hnsw_dist_method": "vector_cosine_ops",
        "ivfflat_probes": 15,
    },
)

if __name__ == '__main__':
    documents = SimpleDirectoryReader("docs").load_data()
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
    )
    
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
    )
    query_engine = index.as_query_engine()
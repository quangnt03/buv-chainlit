import os
import prompt
# Third-party libraries
from typing import List
import chainlit as cl
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.index_store.mongodb import MongoIndexStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from rag import aggregate_answer


@cl.on_chat_start
async def on_chat_start():
    # Initialize the PostgreSQL vector store with hybrid search configuration
    vector_store = PGVectorStore.from_params(
        database=os.getenv("POSTGRES_DB"),
        host=os.getenv("POSTGRES_HOST"),
        password=os.getenv("POSTGRES_PASSWORD"),
        port=os.getenv("POSTGRES_PORT"),
        user=os.getenv("POSTGRES_USER"),
        table_name="vector_store",
        embed_dim=1536,  # OpenAI embedding dimension
        hybrid_search=True,
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 300,
            "hnsw_dist_method": "vector_cosine_ops",
            "ivfflat_probes": 15,
        },
    )
    
    # Initialize the embedding model using OpenAI's API key
    embedding_model = OpenAIEmbedding(api_key=os.getenv("OPEN_API_KEY"))
    doc_store = MongoDocumentStore.from_uri(os.getenv('MONGO_URL'), db_name='doc')
    index_store = MongoIndexStore.from_uri(os.getenv('MONGO_URL'), db_name='index')
    
    storage_context = StorageContext.from_defaults(
        docstore=doc_store,
        index_store=index_store,
        vector_store=vector_store
    )
    
    # Create a vector store index using the configured vector store and embedding model
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embedding_model
    )
    
    vector_retriever = index.as_retriever(
        vector_store_query_mode="default",
        similarity_top_k=5,
        vector_store_kwargs={
            "ivfflat_probes": 15,
            "hnsw_ef_search": 300
        },
    )
    text_retriever = index.as_retriever(
        vector_store_query_mode="sparse",
        similarity_top_k=3,  # interchangeable with sparse_top_k in this context
    )
        
    # Initialize the LLM (GPT-4o) using the OpenAI API key
    llm = OpenAI(
        model='gpt-4o', 
        api_key=os.getenv('OPEN_API_KEY'),
    )
    
    retriever = QueryFusionRetriever(
        [vector_retriever, text_retriever],
        similarity_top_k=5,
        mode=FUSION_MODES.DIST_BASED_SCORE,
        use_async=False,
    )

    response_synthesizer = CompactAndRefine()
    hybrid_query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
    )
    
    # hyde
    query_transform = HyDEQueryTransform(include_original=True)
    query_engine = TransformQueryEngine(
        query_engine=hybrid_query_engine,
        query_transform=query_transform,
    )
    
        
    # Create chat and query engines from the index
    chat_store = SimpleChatStore()
    chat_memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000,
        chat_store=chat_store,
        chat_store_key="user1",
    )
    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=query_engine,
        llm=llm,
        memory=chat_memory,
    )
    # Save the engines in the user session for later use.
    # Also initialize an empty chat history.
    cl.user_session.set('chat_history', [
        ChatMessage(role=MessageRole.SYSTEM, content=prompt.SYSTEM_PROMPT)
    ])
    cl.user_session.set("chat_engine", chat_engine)
    cl.user_session.set("query_engine", query_engine)


@cl.on_message
async def reply(message: cl.Message):
    # Retrieve or initialize chat history
    chat_history: List[ChatMessage] = cl.user_session.get("chat_history", [])
    chat_history.append(
        ChatMessage(role=MessageRole.USER, content=message.content)
    )
    
    chat_engine: BaseChatEngine = cl.user_session.get("chat_engine")
    
    # Generate the full response at once
    response = aggregate_answer(
        chat_engine=chat_engine,
        query=message.content,
        chat_history=chat_history,
    )
    
    # Simulate streaming by sending chunks of the full response
    msg = cl.Message(content='')
    for token in response.response_gen:
        await msg.stream_token(token)

    response_text = response.response
    
    chat_history.append(ChatMessage(role=MessageRole.ASSISTANT, content=response_text))
    cl.user_session.set("chat_history", chat_history)
    msg.content = response_text
    await msg.update()
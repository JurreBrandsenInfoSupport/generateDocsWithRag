# rag.py
from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import UnstructuredFileLoader
import logging
from typing import List
from langchain_core.documents import Document

import documentLoader as t;
import chunker as c;

set_debug(True)
set_verbose(True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RagDocumentationGenerator:
    """A class for handling .cs files ingestion and question answering using RAG."""

    def __init__(self, llm_model: str = "deepseek-r1:latest", embedding_model: str = "mxbai-embed-large"):
        """
        Initialize the RagDocumentationGenerator instance with an LLM and embedding model.
        """
        self.model = ChatOllama(model=llm_model)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant answering questions based on the uploaded document.
            Context:
            {context}

            Question:
            {question}

            Answer concisely and accurately in three sentences or less.
    	    """
        )
        self.vector_store = None
        self.retriever = None

    def ingest(self, file_path: str, debug: bool = False):
        """
        Ingests a C# file, processes its syntax tree, chunks its content,
        and stores embeddings in the vector store.
        """
        logger.info(f"Starting ingestion for file: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        try:
            tree, language_name = t.parse_tree(file_path, content)
            logger.info(f"Parsed file: {file_path}, Language detected: {language_name}")

            chunks = c.chunk_node(tree.root_node, content, debug=debug)
            chunk_docs = [Document(page_content=chunk) for chunk in chunks]

            # Debug: Log each chunk before storing
            logger.info(f"Generated {len(chunk_docs)} chunks:")
            for i, doc in enumerate(chunk_docs):
                logger.info(f"Chunk {i+1}: {doc.page_content[:100]}...")  # Show first 100 chars

            self.vector_store = Chroma.from_documents(
                documents=chunk_docs,
                embedding=self.embeddings,
                persist_directory="chroma_db",
            )

            logger.info("Ingestion completed. Document embeddings stored successfully.")

        except ValueError as e:
            logger.error(f"Failed to ingest {file_path}: {e}")


    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2):
        """
        Answer a query using the RAG pipeline.
        """
        if not self.vector_store:
            raise ValueError("No vector store found. Please ingest a document first.")

        if not self.retriever:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": score_threshold, "k": k},
            )

        logger.info(f"Retrieving context for query: {query}")
        retrieved_docs = self.retriever.invoke(query)

        if not retrieved_docs:
            return "No relevant context found in the document to answer your question."

        # Debug: Print retrieved documents
        logger.info(f"Retrieved {len(retrieved_docs)} documents:")
        for i, doc in enumerate(retrieved_docs):
            logger.info(f"Doc {i+1}: {doc.page_content[:200]}")  # Show first 200 chars

        formatted_input = {
            "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
            "question": query,
        }

        # Build the RAG chain
        chain = (
            RunnablePassthrough()  # Passes the input as-is
            | self.prompt           # Formats the input for the LLM
            | self.model            # Queries the LLM
            | StrOutputParser()     # Parses the LLM's output
        )

        logger.info("Generating response using the LLM.")
        return chain.invoke(formatted_input)



    def clear(self):
        """
        Reset the vector store and retriever.
        """
        logger.info("Clearing vector store and retriever.")
        self.vector_store = None
        self.retriever = None

# rag.py
from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
import logging

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
        Initialize the ChaRagDocumentationGeneratortPDF instance with an LLM and embedding model.
        """
        self.model = ChatOllama(model=llm_model)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = ChatPromptTemplate.from_template(
            """
            **System Prompt:**

            "You are an AI assistant designed to generate high-quality Arc42 documentation based on the provided C# code
            files. Arc42 consists of 12 documents: introduction & goals, constraints, context & scope, solution strategy, building block view,
            runtime view, deployment view, crosscutting concepts, architectural decisions, quality requirements, risks & technical debt, glossary.

            Context:
            {context}

            Question:
            {question}
    	    """
        )
        self.vector_store = None
        self.retriever = None

    def code(self):
        return """
                using System;
                using System.Collections.Generic;
                using System.Linq;
                using System.Text;
                using Microsoft.CodeAnalysis;
                using Microsoft.CodeAnalysis.CSharp;

                namespace TopLevel
                {
                    using Microsoft;
                    using System.ComponentModel;

                    namespace Child1
                    {
                        using Microsoft.Win32;
                        using System.Runtime.InteropServices;

                        class Foo { }
                    }

                    namespace Child2
                    {
                        using System.CodeDom;
                        using Microsoft.CSharp;

                        class Bar { }
                    }
                }"""

    def ingest(self, file_path: str):
        """
        Ingest a C# file, split its contents, and store the embeddings in the vector store.
        """
        logger.info(f"Starting ingestion for file: {file_path}")
        # docs = UnstructuredLoader(file_path=file_path).load()
        # chunks = self.text_splitter.split_documents(docs)
        # chunks = filter_complex_metadata(chunks)
        code = code()
        tree, str = t.parse_tree(file_path, code)
        chunks = c.chunk_node(tree.root_node, code)

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="chroma_db",
        )
        logger.info("Ingestion completed. Document embeddings stored successfully.")

    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2):
        """
        Answer a query using the RAG pipeline.
        """
        if not self.vector_store:
            raise ValueError("No vector store found. Please ingest a document first.")

        if not self.retriever:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": k, "score_threshold": score_threshold},
            )

        logger.info(f"Retrieving context for query: {query}")
        retrieved_docs = self.retriever.invoke(query)

        if not retrieved_docs:
            return "No relevant context found in the document to answer your question."

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

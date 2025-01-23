# Long-Context-Rag-with-RAPTOR
# LangChain Expression Language (LCEL) RAG Pipeline

This project demonstrates a Retrieval Augmented Generation (RAG) pipeline using LangChain Expression Language (LCEL) documentation as the knowledge base. It leverages clustering, summarization, and embedding techniques to create a hierarchical structure for efficient retrieval and generation.

## Overview

The pipeline consists of the following steps:

1. **Data Loading:** LCEL documentation is loaded using RecursiveUrlLoader from langchain_community.
2. **Tokenization:** Documents are tokenized using tiktoken to estimate their size.
3. **Embedding:** Sentence-transformers/all-mpnet-base-v2 is used to generate embeddings for each document.
4. **Clustering:** UMAP and Gaussian Mixture Model (GMM) are employed to cluster documents based on their embeddings. This creates a hierarchical structure of clusters.
5. **Summarization:** Google Gemini is used to generate summaries for each cluster at different levels of the hierarchy.
6. **Vectorstore Creation:** Chroma is used to store all texts (original documents and summaries) along with their embeddings, creating a vectorstore for retrieval.
7. **RAG Chain:** A RAG chain is defined using LangChain hub, which takes a question as input, retrieves relevant documents from the vectorstore, and generates an answer using Google Gemini.

## Usage

1. **Install Dependencies:**
Use code with caution
bash !pip install --quiet -U langchain umap-learn scikit-learn langchain_huggingface langchain_community tiktoken langchain-google-genai langchainhub langchain-chroma langchain-anthropic

 
2. **Set Environment Variables:**
Use code with caution
python import os from google.colab import userdata

os.environ["HUGGINGFACEHUB_API_TOKEN"] = userdata.get('HF_TOKEN')
os.environ["GOOGLE_API_KEY"] = userdata.get("Gemini_Api_Key")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = userdata.get("langchai_api_key") 
os.environ["LANGCHAIN_PROJECT"] = "RAPTOR" 


 
3. **Run the code:**
   Execute the code cells in the provided notebook sequentially.

## Example
Use code with caution
python

Ask a question
rag_chain.invoke("How to define a RAG chain? Give me a specific code example.")

 
## Contributing

Contributions are welcome! Please open an issue or submit a pull request.


## Acknowledgements

- LangChain
- Hugging Face
- Google Gemini
- Chroma
- UMAP
- scikit-learn

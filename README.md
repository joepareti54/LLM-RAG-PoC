# llama2-RAG-PoC
Retrieval-Augmented Generation using Llama 2 (7B) optimized for low GPU memory

# Llama2 RAG Minimal

A minimal, memory-efficient implementation of Retrieval-Augmented Generation using Llama 2 (7B) optimized for consumer GPUs with limited VRAM (6GB+).

## Features
- Two-step RAG pipeline with document summarization
- 4-bit quantization for memory efficiency
- Semantic search using FAISS and SentenceTransformers
- PDF document support

## Requirements
- Python 3.8+
- CUDA-capable GPU with 6GB+ VRAM
- Hugging Face account with access to Llama 2 to acquire the credentials (token)
- your own pdf files. The code is adapted for colab but can be easily changed

## Setup
- Get your Hugging Face token from https://huggingface.co/settings/tokens
- In Colab, update this cell:

import os
os.environ['HUGGINGFACE_TOKEN'] = 'your-token-here'  # Replace with your actual token

- modify the storage path as described in the code


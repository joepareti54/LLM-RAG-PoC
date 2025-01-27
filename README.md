# llama2-RAG-PoC
Retrieval-Augmented Generation using Llama 2 (7B) optimized for low GPU memory

A minimal, memory-efficient implementation of Retrieval-Augmented Generation using Llama 2 (7B) optimized for consumer GPUs with limited VRAM (6GB+).
The code has been tested on an Ubuntu PC with NVIDIA GeForce RTX 2060, and in Colab using T4.
This initial release targets Colab users.

## Features
- Two-step RAG pipeline with document summarization
- 4-bit quantization for memory efficiency
- Semantic search using FAISS and SentenceTransformers
- PDF document support


## Requirements
- Python 3.8+
- CUDA-capable GPU with 6GB+ VRAM
- Hugging Face account with access to Llama 2 to acquire the credentials (token)
- your own pdf files. The code is adapted for Colab. If you want to perate on a local system you may replace the last cell with:
  
```def main():
    from google.colab import drive
    drive.mount('/content/drive')
    
    directory_path = r"/content/drive/My Drive/All_Finance_PDF_files_old/"
    rag = SimpleRAG(directory_path)
    rag.load_pdfs()
    rag.init_embeddings()
    rag.create_index()
    rag.init_llama()

    while True:
        query = input("\nEnter query (or 'quit'): ").strip()
        if query.lower() == 'quit':
            break
        rag.run_query(query, k=3)

if __name__ == "__main__":
    main() 
```

## Setup

- Get your Hugging Face token from https://huggingface.co/settings/tokens
- update this cell:
  
```
import os
os.environ['HUGGINGFACE_TOKEN'] = 'your-token-here'  # Replace with your actual token
```

- modify the storage path as described in the code 

## Maintenance

Refer to this [document](https://docs.google.com/document/d/1pswnEtkYJGR1wkOdShzRw3Sqit-no3POa_v2abXk7uU/edit?usp=sharing)




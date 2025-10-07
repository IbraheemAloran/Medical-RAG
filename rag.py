
!pip install pinecone pypdf langchain langchain[community] streamlit

!pip install langchain_huggingface

!pip install pinecone_text

!pip install langchain-pinecone

!pip install langchain_ollama

import os
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

index_name = "medical-rag"
pc = Pinecone(api_key="*******")

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),

    )

embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

def create_sparse_encoder(doc_chunks):
  text = []
  for doc in doc_chunks:
    text.append(doc.page_content)
  bm25 = BM25Encoder().default()
  bm25.fit(text)
  bm25.dump('bm25_vlaues.json')
  return bm25

from langchain.schema import Document
def data_ingest():
  loader = PyPDFDirectoryLoader('data')
  documents = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
  docs = text_splitter.split_documents(documents)

  docs = [
        Document(page_content="", metadata={"context": doc.page_content, **doc.metadata})
        for doc in docs
    ]

  return docs

pc.Index(index_name)

os.environ["PINECONE_API_KEY"]="*********"

print(os.environ["PINECONE_API_KEY"])

vector_store = PineconeVectorStore.from_documents(documents=data_ingest(), embedding=embeddings, index_name=index_name)

sparser = create_sparse_encoder(data_ingest())

from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # max_new_tokens=10,
)
hf = HuggingFacePipeline(pipeline=pipe)

prompt_template = """
Answer the question based only on the provided context. Use only the information in the context.
If the answer or information is not in the context, say I don't know. Do not make up answers

Context:
{context}

Questions:
{question}

Answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=sparser, index=pc.Index(index_name), top_k=4)

def get_response(llm, retriever, query):
  qa = RetrievalQA.from_chain_type(llm=llm,
                                   chain_type='stuff',
                                   retriever=retriever,
                                   return_source_documents=True,
                                   chain_type_kwargs={"prompt": prompt})
  return qa({"query": query})['result']

get_response(hf, retriever, "what is Avatrombopag")


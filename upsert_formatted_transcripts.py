from langchain_pinecone import PineconeVectorStore
from langchain_together import TogetherEmbeddings
from langchain_core.documents import Document

from pinecone import Pinecone

from dotenv import load_dotenv
import os
import json
from typing import Iterable

from pinecone_utils import load_index

# Load environment variables
load_dotenv()

# Setup API Keys
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "cs224v-lecturebot"

# Configure Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = load_index(pc, index_name=INDEX_NAME)

# initialize PineconeVectorStore and embeddings
together_embedding = TogetherEmbeddings(
    model="togethercomputer/m2-bert-80M-8k-retrieval"
)
vectorstore = PineconeVectorStore(index, embedding=together_embedding, text_key="text")


# Function to load documents from JSONL
def load_docs_from_jsonl(file_path) -> Iterable[Document]:
    array = []
    with open(file_path, "r") as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(
                page_content=data["content"],
                metadata=data["block_metadata"],
            )
            array.append(obj)
    return array


for i in range(1, 19):
    docs = load_docs_from_jsonl(
        f"formatted_lecture_transcriptions/lecture{i}_transcript.jsonl"
    )
    ids = [doc.metadata["id"] for doc in docs]

    vectorstore.add_documents(documents=docs, ids=ids)
    print(f"Lecture {i} processed and added to Pinecone index.")
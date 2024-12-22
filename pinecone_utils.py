from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

import time


def load_index(pc: Pinecone, index_name: str):
    index_name = "cs224v-lecturebot"

    # check if index exists
    try:
        pc.describe_index(index_name)
    except:
        print(f"Index {index_name} does not exist. Creating index...")
        pc.create_index(
            name=index_name,
            dimension=768,  # BERT embeddings have 768 dimensions. This needs to be updated if we use a different embeddings model.
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )

        # wait for index to be initialized
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    # Describe Index
    index = pc.Index(index_name)
    return index

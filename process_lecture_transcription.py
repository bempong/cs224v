from langchain_pinecone import PineconeVectorStore
from langchain_together import TogetherEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pinecone.grpc import PineconeGRPC as Pinecone

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
            obj = Document(**data)
            array.append(obj)
    return array


# Function to process a single lecture.
# transcript file --> auto chapter documents
def process(transcript_file):
    # Load the doc
    docs = load_docs_from_jsonl(transcript_file)
    doc = docs[0]

    # Document title
    lecture_number = int(
        transcript_file.split("/")[-1].split("_transcript")[0].split("lecture")[-1]
    )
    document_title = f"CS224V Lecture {lecture_number}"

    blocks = []

    # Grab auto chapters, put each a list as a document
    chapters = doc.metadata["chapters"]
    for chapter in chapters:
        section_title = document_title + " > Chapter Summaries > " + chapter["gist"]
        block = {
            "document_title": document_title,
            "section_title": section_title,
            "content": f"Title: {section_title}\n\nContent: {chapter['summary']}",
            "block_metadata": {
                "id": section_title.replace(" ", "_"),
                "document_type": "chapter summary",
                "lecture_number": lecture_number,
                "start_ms": chapter["start"],
                "end_ms": chapter["end"],
            },
        }
        blocks.append(block)

    # chunk the main transcript into blocks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    # grab time stamps for each split
    next_words_start_i = 0
    whole_text = doc.page_content

    for i, split in enumerate(
        all_splits[:-1]
    ):  # do all but last, since it will throw an error
        char_start_i = split.metadata["start_index"]
        split_text = split.page_content

        num_words_in_split = len(split_text.split())

        start_time_ms = doc.metadata["words"][next_words_start_i]["start"]
        end_time_ms = doc.metadata["words"][
            next_words_start_i + num_words_in_split - 1
        ]["end"]

        next_char_start_i = all_splits[i + 1].metadata["start_index"]
        next_words_start_i += len(whole_text[char_start_i:next_char_start_i].split())

        section_title = (
            document_title
            + " > Transcript > "
            + str(start_time_ms)
            + " ms - "
            + str(end_time_ms)
            + " ms"
        )
        block = {
            "document_title": document_title,
            "section_title": section_title,
            "content": f"Title: {section_title}\n\nContent: {split_text}",
            "block_metadata": {
                "id": section_title.replace(" ", "_"),
                "document_type": "transcript",
                "lecture_number": lecture_number,
                "start_ms": start_time_ms,
                "end_ms": end_time_ms,
            },
        }
        blocks.append(block)

    # handle last split seperately
    start_time_ms = doc.metadata["words"][next_words_start_i]["start"]
    end_time_ms = doc.metadata["words"][-1]["end"]
    split_text = all_splits[-1].page_content
    section_title = (
        document_title
        + " > Transcript > "
        + str(start_time_ms)
        + " ms - "
        + str(end_time_ms)
        + " ms"
    )
    block = {
        "document_title": document_title,
        "section_title": section_title,
        "content": split_text,
        "block_metadata": {
            "id": section_title.replace(" ", "_"),
            "document_type": "transcript",
            "lecture_number": lecture_number,
            "start_ms": start_time_ms,
            "end_ms": end_time_ms,
        },
    }
    blocks.append(block)

    return blocks


# Rewritten to handle blocks
def save_block_to_jsonl(array: Iterable[Document], file_path: str) -> None:
    with open(file_path, "w") as jsonl_file:
        for doc in array:
            jsonl_file.write(
                json.dumps(doc) + "\n"
            )  # use json.dumps() to serialize block dictionaries


# process range of audio files into chapter docs --> transcribe chapter docs --> save chapter docs as JSONL
def transcribe(transcript_files, output_dir):
    for transcript_file in transcript_files:
        try:
            blocks = process(transcript_file)

            # Generate output file name
            lecture_number = (
                transcript_file.split("/")[-1]
                .split("_transcript")[0]
                .split("lecture")[-1]
            )
            output_file = f"{output_dir}lecture{lecture_number}_transcript.jsonl"

            # Save the transcript to a JSONL file
            save_block_to_jsonl(blocks, output_file)

            print(f"Processed {transcript_file} into {output_file}")
        except Exception as e:
            print(f"Failed to process {transcript_file}: {e}")


# process docs
docs_to_process = [
    f"raw_lecture_transcriptions/lecture{i}_transcript.jsonl" for i in range(1, 15)
]
transcribe(
    docs_to_process,
    "formatted_lecture_transcriptions/",
)

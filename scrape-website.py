import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_together import TogetherEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from typing import List, Dict
import json
import time
import PyPDF2
import io
import re

# Load environment variables
load_dotenv()

# Setup API Keys
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "cs224v-lecturebot"


def get_all_links(soup: BeautifulSoup, base_url: str) -> set:
    """
    Extracts all links from the page that belong to the CS224V domain.
    """
    links = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]

        # Skip certain file types that we don't want to process
        skip_extensions = [".zip", ".tar", ".gz", ".jpg", ".jpeg", ".png", ".gif"]
        if any(href.lower().endswith(ext) for ext in skip_extensions):
            continue
        if href.startswith("/"):
            # Convert relative URLs to absolute
            full_url = f"https://web.stanford.edu{href}"
        elif href.startswith("http"):
            full_url = href
        else:
            full_url = f"{base_url.rstrip('/')}/{href}"

        # Only include links from the CS224V domain
        if "stanford.edu/class/cs224v" in full_url and "#" not in full_url:
            links.add(full_url)
    return links


def extract_pdf_content(pdf_content: bytes, url: str) -> List[Dict]:
    """
    Extracts text content from a PDF file.
    """
    documents = []
    try:
        # Read PDF from memory
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Extract filename from URL to use as title
        filename = url.split("/")[-1]
        title = filename.replace(".pdf", "").replace("_", " ").replace("-", " ").title()

        # Extract text from each page
        full_text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            if text.strip():
                full_text += f"[Page {page_num + 1}]\n{text}\n\n"

        if full_text.strip():
            documents.append(
                {
                    "title": title,
                    "content": full_text,
                    "url": url,
                    "source": "CS224V Website",
                    "type": "course_pdf",
                    "timestamp": datetime.now().isoformat(),
                }
            )

    except Exception as e:
        print(f"Error processing PDF {url}: {str(e)}")

    return documents


def scrape_page(url: str) -> List[Dict]:
    """
    Scrapes a single page or PDF and returns its content.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Check if the content is a PDF
        content_type = response.headers.get("content-type", "").lower()
        if "application/pdf" in content_type or url.lower().endswith(".pdf"):
            return extract_pdf_content(response.content, url)

        soup = BeautifulSoup(response.text, "html.parser")

        documents = []

        # Process main content
        main_content = soup.find(["main", "body"])
        if main_content:
            sections = main_content.find_all(
                ["section", "div", "article"],
                class_=["container", "section", "content"],
            )

            if (
                not sections
            ):  # If no sections found, treat the whole main content as one section
                sections = [main_content]

            for section in sections:
                # Extract title (if available)
                title_elem = section.find(["h1", "h2", "h3"])
                title = title_elem.text.strip() if title_elem else "Course Content"

                # Extract content
                content = ""
                for elem in section.find_all(["p", "li", "pre", "code"]):
                    content += elem.text.strip() + "\n"

                if content.strip():
                    documents.append(
                        {
                            "title": title,
                            "content": content,
                            "url": url,
                            "source": "CS224V Website",
                            "type": "course_website",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

        return documents
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        return []


def scrape_cs224v_website() -> List[Dict]:
    """
    Scrapes the CS224V course website and all its linked pages, returning a list of documents.
    """
    base_url = "https://web.stanford.edu/class/cs224v/"
    visited_urls = set()
    to_visit = {base_url}
    documents = []

    while to_visit:
        url = to_visit.pop()
        if url in visited_urls:
            continue

        print(f"Scraping: {url}")
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")

            # Get content from current page
            page_documents = scrape_page(url)
            documents.extend(page_documents)

            # Find new links
            new_links = get_all_links(soup, base_url)
            to_visit.update(new_links - visited_urls)

            visited_urls.add(url)

        except Exception as e:
            print(f"Error processing {url}: {str(e)}")

        # Add a small delay to be respectful to the server
        time.sleep(1)

    return documents

    # Process main content
    main_content = soup.find("main")
    if main_content:
        sections = main_content.find_all(
            ["section", "div"], class_=["container", "section"]
        )

        for section in sections:
            # Extract title (if available)
            title_elem = section.find(["h1", "h2", "h3"])
            title = title_elem.text.strip() if title_elem else "Course Content"

            # Extract content
            content = ""
            for p in section.find_all(["p", "li"]):
                content += p.text.strip() + "\n"

            if content.strip():
                documents.append(
                    {
                        "title": title,
                        "content": content,
                        "url": base_url,
                        "source": "CS224V Website",
                        "type": "course_website",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

    return documents


def process_documents(documents: List[Dict]) -> List[Dict]:
    """
    Processes the documents by splitting them into smaller chunks.
    """
    # Different chunk sizes for different document types
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    pdf_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Larger chunks for PDFs to maintain context
        chunk_overlap=300,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    processed_docs = []
    for doc in documents:
        # Choose splitter based on document type
        splitter = pdf_splitter if doc["type"] == "course_pdf" else text_splitter
        splits = splitter.split_text(doc["content"])
        for split in splits:
            processed_docs.append(
                {
                    "title": doc["title"],
                    "content": split,
                    "url": doc["url"],
                    "source": doc["source"],
                    "type": doc["type"],
                    "timestamp": doc["timestamp"],
                }
            )

    return processed_docs


def save_to_jsonl(documents: List[Dict], filename: str):
    """
    Saves the documents to a JSONL file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")


def upload_to_pinecone(documents: List[Dict]):
    """
    Uploads the documents to Pinecone.
    """
    from langchain_core.documents import Document

    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    # Initialize embeddings
    together_embedding = TogetherEmbeddings(
        model="togethercomputer/m2-bert-80M-8k-retrieval", api_key=TOGETHER_API_KEY
    )

    # Initialize vector store
    vectorstore = PineconeVectorStore(
        index, embedding=together_embedding, text_key="text"
    )

    # Prepare documents for upload
    docs_for_upload = []
    for doc in documents:
        # Create a Document object
        langchain_doc = Document(
            page_content=f"Title: {doc['title']}\n\nContent: {doc['content']}",
            metadata={
                "title": doc["title"],
                "url": doc["url"],
                "source": doc["source"],
                "type": doc["type"],
                "timestamp": doc["timestamp"],
            },
        )
        docs_for_upload.append(langchain_doc)

    # Upload to Pinecone in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(docs_for_upload), batch_size):
        batch = docs_for_upload[i : i + batch_size]
        vectorstore.add_documents(batch)
        print(
            f"Uploaded batch {i//batch_size + 1} of {(len(docs_for_upload)-1)//batch_size + 1}"
        )


def load_documents_from_jsonl(filename: str) -> List[Dict]:
    """
    Loads documents from a JSONL file.
    """
    documents = []
    with open(filename) as f:
        for line in f:
            doc = json.loads(line)
            documents.append(doc)
    return documents


def main():
    # Scrape website
    # print("Scraping CS224V website...")
    # documents = scrape_cs224v_website()

    # Process documents
    # print("Processing documents...")
    # processed_docs = process_documents(documents)

    # Save to JSONL
    output_file = "cs224v_website_content.jsonl"
    # print(f"Saving to {output_file}...")
    # save_to_jsonl(processed_docs, output_file)

    # Load and Upload to Pinecone
    print("Uploading to Pinecone...")
    processed_docs = load_documents_from_jsonl(output_file)
    upload_to_pinecone(processed_docs)

    print("Done!")


if __name__ == "__main__":
    main()

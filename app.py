import streamlit as st
import streamlit_nested_layout  # unused as an import, but required for streamlit to support nested layouts

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_together import TogetherEmbeddings
import uuid


from pinecone.grpc import PineconeGRPC as Pinecone

from dotenv import load_dotenv
import os
import json
import time
from datetime import datetime

from pinecone_utils import load_index

# Load environment variables
load_dotenv()

# Setup API Keys
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# App Configuration
title = "CS224V LectureBot 📚"
st.set_page_config(page_title=title, page_icon="🤖")
st.title(title)

# Configure Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "cs224v-lecturebot"
index = load_index(pc, index_name=index_name)

# initialize PineconeVectorStore and embeddings
together_embedding = TogetherEmbeddings(
    model="togethercomputer/m2-bert-80M-8k-retrieval"
)
vectorstore = PineconeVectorStore(index, embedding=together_embedding, text_key="text")

# Initialize ChatOpenAI for TogetherAI
llm = ChatOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY,
    model="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    streaming=True,
)


def ms_to_min_sec(ms):
    minutes = ms // 60000
    seconds = (ms % 60000) // 1000
    return f"{int(minutes):02d}:{int(seconds):02d}"


def format_docs(docs):
    formatted_docs = "\n\n".join(
        [
            json.dumps({"content": doc.page_content, "metadata": doc.metadata})
            for doc in docs
        ]
    )

    citations = []
    for doc in docs:
        if "lecture_number" in doc.metadata:
            # Handle lecture transcript
            citation = {
                "title": f"Lecture {int(doc.metadata['lecture_number'])} ({ms_to_min_sec(doc.metadata.get('start_ms', 0))} - {ms_to_min_sec(doc.metadata.get('end_ms', 0))})",
                "content": doc.page_content,
                "lecture_number": int(doc.metadata["lecture_number"]),
                "start_ms": doc.metadata.get("start_ms", 0),
                "source_type": "lecture",
            }
        else:
            # Handle website content
            citation = {
                "title": f"Website: {doc.metadata.get('title', 'Unknown')}",
                "content": doc.page_content,
                "url": doc.metadata.get("url", ""),
                "source_type": "website",
            }
        citations.append(citation)

    # Sort citations - lectures by number/time, websites alphabetically by title
    citations.sort(
        key=lambda x: (
            (x["lecture_number"], x["start_ms"])
            if x["source_type"] == "lecture"
            else (float("inf"), x["title"])
        )
    )

    return formatted_docs, citations


def rewrite_query(user_query, chat_history):
    template = """
    Given the following conversation history and the latest user query, rewrite the user query to be independently standing and provide enough context for a retrieval system to fetch relevant documents.
    Do not include any additional text or explanations, only the rewritten user query. Keep the response fairly concise.

    Today's date: {current_date}

    Conversation history:
    {chat_history}

    Latest user query:
    {user_query}

    Rewritten user query:
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm

    current_date = datetime.now().strftime("%Y-%m-%d")

    response = chain.invoke(
        {
            "chat_history": "\n\n".join([msg.content for msg in chat_history]),
            "user_query": user_query,
            "current_date": current_date,
        }
    )
    rewritten_query = response.content.strip()
    return rewritten_query


def retrieve_context(user_query, chat_history):
    template = """
    You are a bot that extracts metadata filters from user queries. You are given a user query and you need to extract the metadata filters from the query. You have access to:
    1. Lecture transcripts up to lecture 18
    2. Course website content and documentation

    This is useful for retrieving specific chunks of text from documents based on user queries. If you are not able to extract the metadata filters from the user query, you should return an empty dictionary.
    
    The following schema is used to represent the metadata filters:
        For lecture content:
        {{
            "document_type": str, # one of "transcript", "chapter summary"
            "lecture_number": int, # the lecture number that the chunk of text comes from
            "start_ms": int, # the start time of the chunk of text in milliseconds (for transcripts)
            "end_ms": int # the end time of the chunk of text in milliseconds (for transcripts)
        }}
        
        For website content:
        {{
            "type": str, # one of "course_website", "course_pdf",
        }}
    Do not include any additional filters or properties not specified in the schema.

    You have access to these filter operators:
    {{
      "filters": [
        {{
          "operator": "$eq",
          "description": "Matches vectors with metadata values that are equal to a specified value.",
          "supported_types": ["number", "string", "boolean"]
        }},
        {{
          "operator": "$ne",
          "description": "Matches vectors with metadata values that are not equal to a specified value.",
          "supported_types": ["number", "string", "boolean"]
        }},
        {{
          "operator": "$gt",
          "description": "Matches vectors with metadata values that are greater than a specified value.",
          "supported_types": ["number"]
        }},
        {{
          "operator": "$gte",
          "description": "Matches vectors with metadata values that are greater than or equal to a specified value.",
          "supported_types": ["number"]
        }},
        {{
          "operator": "$lt",
          "description": "Matches vectors with metadata values that are less than a specified value.",
          "supported_types": ["number"]
        }},
        {{
          "operator": "$lte",
          "description": "Matches vectors with metadata values that are less than or equal to a specified value.",
          "supported_types": ["number"]
        }},
        {{
          "operator": "$in",
          "description": "Matches vectors with metadata values that are in a specified array.",
          "supported_types": ["string", "number"]
        }},
        {{
          "operator": "$nin",
          "description": "Matches vectors with metadata values that are not in a specified array.",
          "supported_types": ["string", "number"]
        }},
        {{
          "operator": "$exists",
          "description": "Matches vectors with the specified metadata field.",
          "supported_types": ["boolean"]
        }}
      ]
    }}
    
    For example, given the following user query: "Summarize the first 10 minutes of lecture 1", your response should be:
    {{
      "lecture_number": 1,
      "start_ms": {{"$lte": 600000}},
    }}

    Another example, given the following user query: "Summarize the first half of lecture 1", your response should be:
    {{
      "lecture_number": 1,
      "start_ms": {{"$lte": 2700000}},
    }}

    Another example, given the following user: "What is the summary of the first chapter of lecture 2?", your response should be:
    {{
      "lecture_number": 2
    }}

    If you are not able to extract the metadata filters from the user query or are not fully confident in your extraction, your response should be:
    {{}}

    Extract the metadata filters from the user query below and respond only in JSON format. Do not include any additional text or explanations.
    Today's date: {current_date}

    User query: {user_query}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm

    rewritten_query = rewrite_query(user_query, chat_history)

    response = chain.invoke(
        {
            "user_query": rewritten_query,
            "current_date": datetime.now().strftime("%Y-%m-%d"),
        }
    )
    cleaned_response = response.content.replace("```", "").replace("\\n", "")

    try:
        metadata_filters = json.loads(cleaned_response)
    except json.JSONDecodeError:
        metadata_filters = {}

    try:
        print("Retrieving context with metadata filters:", metadata_filters)
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 20, "filter": metadata_filters},
        )
        retrieved_context = retriever.invoke(rewritten_query)

        if len(retrieved_context) == 0:
            print(
                "No documents retrieved with metadata filters. Using default retrieval."
            )
            raise Exception("No documents retrieved with metadata filters.")
    except:
        print("Error retrieving context. Using default retrieval.")
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 20}
        )
        retrieved_context = retriever.invoke(rewritten_query)

    print(len(retrieved_context))
    formatted_docs, citations = format_docs(retrieved_context)
    return formatted_docs, citations, retrieved_context, metadata_filters


def generate_suggested_questions(user_query, chat_history):
    """
    Generates suggested questions based on the current conversation.
    """
    template = """
    Based on the following conversation history and the latest user query, generate three consise suggested follow-up questions that the user might ask next.
    Make sure the questions are relevant to the context and provide additional value to the user. Also, make them simple questions that can be answered based on the course content. 
    Note that this chatbot only has access to lecture transcripts and metadata, so the suggested questions should be related to the course content. 
    Also the bot does not have access to specific lecture dates, only lecture numbers (up to lecture 18).

    Do not include any additional text or explanations, only the suggested questions, separated by a single newline.

    Today's date: {current_date}

    Conversation history:
    {chat_history}

    Latest user query:
    {user_query}

    Suggested follow-up questions:
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm

    current_date = datetime.now().strftime("%Y-%m-%d")

    response = chain.invoke(
        {
            "chat_history": "\n\n".join([msg.content for msg in chat_history]),
            "user_query": user_query,
            "current_date": current_date,
        }
    )
    suggested_questions = response.content.strip().split("\n\n")
    return suggested_questions


def stream_response(user_query, chat_history):
    """
    Streams response from TogetherAI using LangChain's ChatOpenAI wrapper and generates suggested follow-up questions.
    """
    template = """
    You are a course assistant bot that answers questions based on the course content. You have access to:
    1. Lecture transcripts up to lecture 18
    2. Course website content

    Use the following pieces of context and chat history to answer the question at the end, and respond kindly.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer concise.

    For each sentence in the response, please cite the source of the information inline. If the information is from a lecture, include only the lecture number and the time range in the citation 
    (in minutes and seconds, not milliseconds). 
    If the information is from a website, include only the title of the website in the citation, and a link using markdown. Make sure the citations are easily legible.

    Always say "thanks for asking!" at the end of the answer.

    Today's date: {current_date}

    Chat history: {chat_history}

    User question: {user_question}

    Retrieved context: {retrieved_context}
    """

    # Prepare the prompt
    prompt = ChatPromptTemplate.from_template(template)

    formatted_docs, citations, retrieved_context, metadata_filters = retrieve_context(
        user_query, chat_history
    )

    # Combine prompt and TogetherAI in LangChain pipeline
    chain = prompt | llm | StrOutputParser()

    # Stream response
    current_date = datetime.now().strftime("%Y-%m-%d")

    response_stream = chain.stream(
        {
            "retrieved_context": retrieved_context,
            "user_question": user_query,
            "chat_history": "\n\n".join([msg.content for msg in chat_history]),
            "current_date": current_date,
        }
    )

    response = ""
    for chunk in response_stream:
        response += chunk
        yield response, citations


def set_user_query(user_query):
    st.session_state.user_query = user_query


# Session State for Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hi! I am the CS224V LectureBot. How can I assist you today?")
    ]

# Display Chat History
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# Handle User Input
if "user_query" not in st.session_state:
    st.session_state.user_query = None

initial_suggested_questions = [
    "Summarize the first 10 minutes of lecture 1.",
    "What is the final project presentation?",
    "What does the last lecture cover?",
]

# Show initial suggested questions
user_query = st.chat_input("Type your question here...", key="main_input")
if user_query:
    set_user_query(user_query)

if st.session_state.user_query is None:
    with st.expander("Suggested Questions"):
        for question in initial_suggested_questions:
            st.button(question, on_click=set_user_query, args=[question])

# Process User Input
# If user input or a suggestion is selected, process it
if st.session_state.user_query:
    user_query = st.session_state.user_query
    set_user_query(None)

    st.session_state.chat_history.append(HumanMessage(content=user_query))

    # Display user input
    with st.chat_message("Human"):
        st.markdown(user_query)

    # Stream LLM Response
    with st.chat_message("AI"):
        with st.spinner("Generating response..."):
            response_generator = stream_response(
                user_query, st.session_state.chat_history
            )
            response_placeholder = st.empty()
            response, citations = "", []
            for response, citations in response_generator:
                response_placeholder.write(response)

    # Append AI response to chat history
    st.session_state.chat_history.append(AIMessage(content=response))

    # Generate Suggested Follow-Up Questions
    suggested_questions = generate_suggested_questions(
        user_query, st.session_state.chat_history
    )
    with st.expander("Suggested Follow-Up Questions"):
        for question in suggested_questions:
            st.button(question, on_click=set_user_query, args=[question])

    # Display Citations
    if citations:
        with st.expander("View All Citations"):
            st.write("Here are the citations for the retrieved context:")
            for citation in citations:
                with st.expander(citation.get("title", "Citation")):
                    if citation.get("source_type") == "website" and citation.get("url"):
                        st.markdown(
                            f"**Source:** [{citation['url']}]({citation['url']})"
                        )
                    st.write(citation.get("content", ""))

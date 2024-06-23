import re

import streamlit as st
from bs4 import BeautifulSoup
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.utilities.requests import (
    JsonRequestsWrapper,
    TextRequestsWrapper,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents.base import Document
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import ToolException
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from markdownify import markdownify as md
from openai import embeddings


def setup():
    """Configure the initial screen layout and the initial settings for the agent of the application."""
    global embeddings
    global chat_disabled
    global agent_executor

    st.title("Quarkus Finder")
    # Initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = StreamlitChatMessageHistory(key="chat_messages")

    # Display chat messages from history on app rerun
    for chat in st.session_state.history.messages:
        st.chat_message(chat.type).write(chat.content)

    # Configure the sidebar.
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    model = st.sidebar.selectbox(
        label="Model", options=("gpt-3.5-turbo", "gpt-4o", "gpt-4-turbo"), index=None
    )
    temperature = st.sidebar.slider(
        label="Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        format="%.1f",
    )
    st.sidebar.button(
        label="Clear History",
        help="Clicking this button will start a new chat.",
        type="primary",
        on_click=st.session_state.history.clear,
    )

    # Setup the LangChain Agent.
    if not openai_api_key.startswith("sk-") or model is None:
        chat_disabled = True
        st.warning("Please enter your OpenAI API key and choose a model!", icon="⚠")
    else:
        chat_disabled = False
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            # max_tokens=None,
            # timeout=None,
            # max_retries=2,
            api_key=openai_api_key,
        )
        tools = [search_quarkus_guides, get_guide_contents]
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are an assistant dedicated to solving users' issues related to Quarkus.
                    You can search for the latest Quarkus guides and refer to their content to provide useful information to users.
                    While the guides are written in English, you must respond in the user's preferred language.
                    If you find multiple important guides, review their content and provide a response to the user.
                    """,
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        agent = create_tool_calling_agent(llm, tools, prompt=prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", openai_api_key=openai_api_key
        )


def main():
    """Display chat messages and interact with the LangChain Agent."""
    # React to user input
    if input_msg := st.chat_input("Send message via 🦜🔗", disabled=chat_disabled):
        # Add user message to chat history
        st.session_state.history.add_message(HumanMessage(content=input_msg))
        # Display user message in chat message container
        with st.chat_message("humman"):
            st.markdown(input_msg)

        # Display assistant response in chat message container
        with st.chat_message("ai"):
            st_callback = StreamlitCallbackHandler(st.container())
            response = agent_executor.invoke(
                {"input": input_msg, "chat_history": st.session_state.history.messages},
                {"callbacks": [st_callback]},
            )

            st.markdown(response["output"])
        # Add assistant response to chat history
        st.session_state.history.add_message(AIMessage(content=response["output"]))


@tool
def search_quarkus_guides(keywords: list) -> dict:
    """
    A tool to search for guides on the official Quarkus website.

    This function constructs a search query from a list of keywords
    (which must be in English), sends a GET request to the Quarkus.io Search
    (https://search.quarkus.io/api/guides/search), and returns the search results.

    Args:
        keywords (list): A list of strings representing search keywords.
                         Note that keywords must be in English.

    Returns:
        dict: The search results from the Quarkus.io Search, returned as a dictionary.
    """
    requests = JsonRequestsWrapper()
    query = "+".join(keywords)
    ret = requests.get(f"https://search.quarkus.io/api/guides/search?q={query}")

    return ret


@tool
def get_guide_contents(urls: list[str], question: str) -> list[Document]:
    """
    Retrieves sections related to a topic from guide pages on the Quarkus website.

    This function fetches the content of Quarkus guides given their URLs, parses
    the HTML to extract the main guide content, splits it into sections based
    on predefined header tags, and searches the processed content for sections
    related to the provided topic.

    Args:
        urls (list[str]): A list of URLs of the guides. Each URL must be from
                          the quarkus.io website.
        question (str): A sentence or phrase in English detailing the specific information or subject of interest.

    Raises:
        ToolException: If any provided URL is not from the quarkus.io website.

    Returns:
        str: The sections related to the question.
    """

    def get_contents(url: str) -> str:
        if not url.startswith("https://quarkus.io/"):
            raise ToolException(f"Error: {url} is not from the quarkus.io website.")

        # Access the guide page and extract the content.
        page_text = TextRequestsWrapper().get(url)
        soup = BeautifulSoup(page_text, "html.parser")
        guide_contents = soup.find("div", {"class": "guide"})
        html_text = str(guide_contents)

        # Convert to Markdown and remove any unnecessary strings.
        markdown = md(html_text)
        markdown = re.sub(r"\n\s*\n+", "\n\n", markdown.strip())
        markdown = re.sub(r"\[Edit this Page\]\(.*\)\n", "", markdown)

        # Split the Markdown into chunks based on sections and token count.
        headers_to_split_on = [
            ("#", "Section"),
            ("##", "Section"),
            ("###", "Section"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(markdown)

        chunk_size = 4000
        chunk_overlap = 50
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        chunks = text_splitter.split_documents(md_header_splits)

        cached.add(url)
        return chunks

    if "cached" not in st.session_state:
        st.session_state.cached = set()
    cached = st.session_state.cached

    # Collect contents from URLs that are not in the cache
    chunks = [
        doc
        for url in urls
        if url not in cached  # Filter out cached URLs
        for doc in get_contents(url)  # Retrieve contents from each URL
    ]

    # Store the chunks in the vector store.
    if chunks != []:
        if "vectorstore" not in st.session_state:
            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)
        else:
            st.session_state.vectorstore.add_documents(documents=chunks)

    vectorstore = st.session_state.vectorstore

    # Search for relevant sections from the vector store.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    sections = retriever.invoke(question)
    # Concatenate the markdonw sections.
    text = ""
    for section in sections:
        text += f"""## {section.metadata.get("Section", "title")}

{section.page_content}

"""
    return text


if __name__ == "__main__":
    setup()
    main()

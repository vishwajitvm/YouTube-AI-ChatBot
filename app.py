import streamlit as st
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# Function to extract video ID
def extract_video_id(youtube_url: str) -> str:
    """
    Extract the video ID from a full YouTube URL.
    Supports formats:
    - https://www.youtube.com/watch?v=xxxx
    - https://youtu.be/xxxx
    """
    parsed_url = urlparse(youtube_url)
    if "youtu.be" in parsed_url.netloc:
        return parsed_url.path.strip("/")
    elif "youtube.com" in parsed_url.netloc:
        params = parse_qs(parsed_url.query)
        return params.get("v", [None])[0]
    else:
        return None

# Streamlit UI
st.title("🎥 YouTube Video QA Assistant")
st.markdown(
    """
    Enter a **YouTube video URL**.  
    Example: `https://www.youtube.com/watch?v=HAnw168huqA` or `https://youtu.be/HAnw168huqA`.  
    ⚠️ The video must have **captions enabled** (auto-generated or uploaded).
    """
)

youtube_url = st.text_input("YouTube Video URL:", value="https://www.youtube.com/watch?v=HAnw168huqA")
question = st.text_input("Ask your question about the video:")

video_id = extract_video_id(youtube_url)

if not video_id:
    st.error("❌ Could not extract video ID. Please check the URL format.")
else:
    # Common prompt template
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        Provide a detailed and thorough answer. Include examples or elaboration where possible.

        {context}
        Question: {question}
        """,
        input_variables=["context", "question"],
    )

    if st.button("Get Answer"):
        try:
            with st.spinner("Fetching transcript and building knowledge base..."):
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
                transcript = " ".join(chunk["text"] for chunk in transcript_list)

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.create_documents([transcript])

                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vector_store = FAISS.from_documents(chunks, embeddings)

                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

                retrieved_docs = retriever.invoke(question)
                context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
                final_prompt = prompt.invoke({"context": context_text, "question": question})

                answer = llm.invoke(final_prompt)

            st.success("Answer:")
            st.write(answer.content)

        except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript):
            st.error("❌ No captions available for this video. Please try a different one with subtitles enabled.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

    if st.button("Summarize Entire Video"):
        try:
            with st.spinner("Fetching transcript and summarizing..."):
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
                transcript = " ".join(chunk["text"] for chunk in transcript_list)

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.create_documents([transcript])

                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vector_store = FAISS.from_documents(chunks, embeddings)

                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

                def format_docs(retrieved_docs):
                    return "\n\n".join(doc.page_content for doc in retrieved_docs)

                parallel_chain = RunnableParallel({
                    'context': retriever | RunnableLambda(format_docs),
                    'question': RunnablePassthrough()
                })

                parser = StrOutputParser()
                main_chain = parallel_chain | prompt | llm | parser

                summary = main_chain.invoke("Can you summarize the video")

            st.success("Summary:")
            st.write(summary)

        except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript):
            st.error("❌ No captions available for this video. Please try a different one with subtitles enabled.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

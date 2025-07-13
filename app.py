from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
# from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage)
from langchain_google_genai import ChatGoogleGenerativeAI, embeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)


########################################
#Step 1a - Indexing (Document Ingestion)
video_id = "Gfr50f6ZBvo" # only the ID, not full URL
try:
    # If you don’t care which language, this returns the “best” one
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    # print("transcript_list:", transcript_list)
    
    # Flatten it to plain text
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    # print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")
    


####################################
#Step 1b - Indexing (Text Splitting)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
# print(len(chunks))
# print(chunks[100])



##########################################################################
#Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_documents(chunks, embeddings)

# print("Vector store created with", len(vector_store.index_to_docstore_id), "documents.")
# print("all document in vector store:", vector_store.index_to_docstore_id)
# print(vector_store.get_by_ids(['7abd3578-4021-40b0-b28f-e8189712a948']))



#########################################
########################################
#Step 2 - Retrieval
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
# retriever.invoke('What is deepmind')


#########################################
########################################
#Step 3 - Augmentation
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)
# print("retrieved_docs:", retrieved_docs)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
# print("context_text:" , context_text)
final_prompt = prompt.invoke({"context": context_text, "question": question})



##############################################
##############################################
#Step 4 - Generation
answer = llm.invoke(final_prompt)
print(answer.content)


###############################################
###############################################
#Building a Chain
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text


parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

# parallel_chain.invoke('who is Demis')

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

final_result_of_final_question = main_chain.invoke('Can you summarize the video')

print("Final Result of Final Question:", final_result_of_final_question)
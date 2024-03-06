from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.vectorstores.faiss import FAISS
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate

model_path=""
embeddings = LlamaCppEmbeddings(model_path=model_path)

def create_vector_from_yt_url(url:str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(url, language="pt") # (url, language="pt")
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embedding=embeddings)
    return db

def get_response_from_query(db, query, k=1):
    docs = db.similarity_search(query, k=k)
    docs_page_content = "".join([d.page_content for d in docs])

    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.75,
        max_tokens=1000,
        n_batch=1000,
        n_ctx=2048,
        verbose=False,  # Verbose is required to pass to the callback manager
    )

    template = ChatPromptTemplate.from_messages(
    [
        ("user", 
         """You are an assistant who answers questions about YouTube videos based on the video transcript
        
            Answer this question: {question}
            Based in this transcriptions: {docs}

            Use only informations in the trascriptions, if you dont know the answer, reply with "I dont know, sorry."""   
        ),
    ]
)
    chain = LLMChain(llm=llm, prompt=template, output_key="answer")
    response = chain({"question":query, "docs":docs_page_content})

    return response, docs


if(__name__ == "__main__"):
    db = create_vector_from_yt_url()
    response, docs = get_response_from_query(
        db,
        "Question here..."
    )

    print(response)
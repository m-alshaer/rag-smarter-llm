from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import os

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
MODEL = "gpt-3.5-turbo"
model = ChatOpenAI(api_key=OPENAI_API_KEY, model=MODEL)
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

def define_chain(retriever):
    template = """
    Answer the question based on the context below. If you can't
    answer the question, reply "I don't know".
    
    Context: {context}
    
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    # prompt.format(context="Here is some context", question="Here is a question")
    parser = StrOutputParser()

    chain = (
            {
                "context": itemgetter("question") | retriever,
                "question": itemgetter("question")
            }
            | prompt
            | model
            | parser
    )

    return chain

def get_retriever(docs):
    vectorstore = DocArrayInMemorySearch.from_documents(docs, embedding=embeddings)
    return vectorstore.as_retriever()

def get_model(docs):
    retriever = get_retriever(docs)
    chain = define_chain(retriever)
    return chain
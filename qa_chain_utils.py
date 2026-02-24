from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI


def build_qa_chain(vectorstore, openai_api_key):

    llm = ChatOpenAI(
        model="gpt-4o-mini", # use gpt-4o-mini
        openai_api_key=openai_api_key, #
        temperature=1
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    return qa_chain

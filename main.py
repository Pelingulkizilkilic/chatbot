from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings  # Güncellenmiş import
from langchain_openai import ChatOpenAI  # Güncellenmiş import
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Modeli başlat
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Dosyayı yükle
loader = TextLoader("C:/Users/p/Desktop/document.txt")
docs = loader.load()

# Metni parçalara ayırma
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Vektör veritabanı oluşturma
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieval işlemi için retriever
retriever = vectorstore.as_retriever()

# Prompt şablonunu yükleme
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG zincirini oluşturma
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Ana çalışma döngüsü
if __name__ == "__main__":
    while True:
        user_input = input("Sana nasıl yardımcı olabilirim? ")
        response = rag_chain.invoke(user_input)
        print("Cevap:", response)

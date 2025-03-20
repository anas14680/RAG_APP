import glob
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts.chat import MessagesPlaceholder


# Use glob to find all PDF files in the folder
pdf_files = glob.glob("docs/*.pdf")

# parse pdf files and store them in all_docs
all_docs = []


for pdf in pdf_files:
    docs_loader = PyPDFLoader(pdf)
    docs = docs_loader.load()
    all_docs.extend(docs)

# create documents into chunks with overlaps, as we cannot feed the whole doc at once
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_docs = text_splitter.split_documents(all_docs)


# Generate embeddings for the text
embed = OllamaEmbeddings(model='nomic-embed-text')
db = Chroma.from_documents(final_docs, embed)

# initiate chat history
chat_history = [] 


# initialize a deepseek model ( use for chat history)
llm= Ollama(model='deepseek-r1')

hrtr_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name='chat_history'),
    ("human", "{input}"),
    ("human", "Given the above conversation and the human input, generate a search query to look up in order to get relevant information to the conversation."),
])

# prompt for creating new question based on history
prompt_messages = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's question based on the context: {context} and below chat."),
    MessagesPlaceholder(variable_name='chat_history'),
    ("human", "{input}")
])

# set up a chain document that passes the  prompt (formatted into llm)
document_chain = create_stuff_documents_chain(llm, prompt_messages)




# Create a retriever with chroma db at the back end and form a chain
retreiver = db.as_retriever()
hrtr = create_history_aware_retriever(llm, retreiver, hrtr_prompt)
rtr_chain = create_retrieval_chain(hrtr, document_chain)



# Run chatbot
while True:
    user_input = input('You: ')
    print(user_input)
    if user_input.lower() == 'exit':
        break
    response = rtr_chain.invoke({
        'input': user_input,
        'chat_history': chat_history
    })['answer']

    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))

    print('Assisstant:  ', response)

# nc = [
#     Document(page_content="My name is Anas")
# ]
 
# ni = [
#     Document(page_content="What is my name?")
# ]


# document_chain.invoke({
#     'context':nc,
#     'input': ni
# })

# formatted_prompt = chat_prompt.format(context="HRV is Measurable", input="What is it?")
# print(formatted_prompt)



# Generate embeddings for the text
# embed = OllamaEmbeddings(model='nomic-embed-text')
# db = Chroma.from_documents(split_texts, embed)

# db.similarity_search_with_score('What is HRV', k=1)


import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.9,
    api_key=api_key
)

prompt_de_sugestao = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um assistente de viagens que sugere destinos brasileiros. Apresenten-se como Sr. Passeio."),
        ("placeholder", "{historico}"),
        ("human", "{query}"),
    ]
)

chain = prompt_de_sugestao | model | StrOutputParser()

memory = {}
sessao = "langchain"

def historico_por_sessao(sessao : str):
    if sessao not in memory:
        memory[sessao] = InMemoryChatMessageHistory()
    return memory[sessao]
   

list_questions = [
    "Quero visitar um lugar no Brasil, famoso por praias e cultura. Pode sugerir?",
    "Qual a melhor época do ano pra ir?"
]

chain_with_memory = RunnableWithMessageHistory(
    runnable = chain,
    get_session_history = historico_por_sessao,
    input_messages_key = "query",
    history_messages_key = "historico" 
)

for question in list_questions:
    answer = chain_with_memory.invoke(
        {
            "query": question
        },
        config={"session_id": sessao}
    ),
    print("Usuário:" , question),
    print("IA:" , answer, "\n")
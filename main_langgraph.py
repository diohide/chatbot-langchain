from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from typing import TypeDict, Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
import asyncio
import os

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.9,
    api_key=os.getenv("OPENAI_API_KEY")
)

prompt_consultor_praia = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se como SRA. Praia. Você é uma especialista de viagens para praia."),
        ("human", "{query}"),
    ]
)

prompt_consultor_montanha = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se como SR. Montanha. Você é uma especialista de viagens para montanha e atividades radicais."),
        ("human", "{query}"),
    ]
)

class Rota(TypedDict):
    destino: Literal["praia", "montanha"]

prompt_roteador = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda apenas com 'praia' ou 'montanha'"),
        ("human", "{query}"),
    ]
)

roteador = prompt_roteador | model.with_structured_output(Rota)


chain_praia = prompt_consultor_praia | model | StrOutputParser()
chain_montanha = prompt_consultor_montanha | model | StrOutputParser()

class Estado(TypedDict):
    query: str
    destino: Rota
    response: str

async def no_roteador(estado: Estado, config=RunnableConfig):
    return {"destino": await roteador.ainvoke({"query": estado["query"]}, config)}

async def no_praia(estado: Estado, config=RunnableConfig):
    return {"response": await chain_praia.ainvoke({"query": estado["query"]}, config)}

async def no_montanha(estado: Estado, config=RunnableConfig):
    return {"response": await chain_montanha.ainvoke({"query": estado["query"]}, config)}

def escolher_no(estado: Estado)-> Literal["praia", "montanha"]:
    return "praia" if estado["destino"]["destino"] == "praia" else "montanha"

grafo = StateGraph(Estado)
grafo.add_node("rotear", no_roteador)
grafo.add_node("praia", no_praia)
grafo.add_node("montanha", no_montanha)

grafo.add_edge(START, "rotear")
grafo.add_conditional_edges("rotear", escolher_no)
grafo.add_edge("praia", END)
grafo.add_edge("montanha", END)

app = grafo.compile()

async def main():
    resposta = await app.ainvoke(
        {
            "query": "Quero visitar um lugar no Brasil, famoso por praias e cultura. Pode sugerir?"
        }
    )
    print(resposta["response"])

asyncio.run(main()) 
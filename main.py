from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.globals import set_debug
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

class Destino(BaseModel):
    cidade: str = Field("A cidade recomendada para visitar")
    motivo: str = Field("O motivo pelo qual essa cidade é recomendada")

class Restaurantes(BaseModel):
    cidade: str = Field("A cidade recomendada para visitar")
    restaurantes: str = Field("Uma lista de restaurantes recomendados nessa cidade")

parseador_destino = JsonOutputParser(pydantic_object=Destino)
parseador_restaurantes = JsonOutputParser(pydantic_object=Restaurantes)

prompt_cidade = PromptTemplate(
    template="""
    Sugira uma cidade dado o meu interesse por {interesse}.
    {formato_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formato_de_saida" : parseador_destino.get_format_instructions() }
)

prompt_restaurantes = PromptTemplate(
    template="""
    Sugira uma lista de restaurantes na cidade {cidade}.
    {formato_de_saida}
    """,
    partial_variables={"formato_de_saida" : parseador_restaurantes.get_format_instructions() }
)

prompt_cultura = PromptTemplate(
    template= "Sugira atividades culturais para fazer na cidade {cidade}."
)

modelo = ChatOpenAI(
    model="gpt-3.5-turbo", 
    temperature=0.9, 
    api_key=api_key
)

chain_1 = prompt_cidade | modelo | parseador_destino
chain_2 = prompt_restaurantes | modelo | parseador_restaurantes
chain_3 = prompt_cultura | modelo | StrOutputParser()

chain = ()


response = chain.invoke(
    {
        "interesse": "praias"
    }
)
print(response)
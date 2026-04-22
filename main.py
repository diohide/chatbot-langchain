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

parseador = JsonOutputParser(pydantic_object=Destino)

prompt_cidade = PromptTemplate(
    template="""
    Sugira uma cidade dado o meu interesse por {interesse}.
    {formato_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formato_de_saida" : parseador.get_format_instructions() }
)

modelo = ChatOpenAI(
    model="gpt-3.5-turbo", 
    temperature=0.9, 
    api_key=api_key
)

chain = prompt_cidade | modelo | parseador


response = chain.invoke(
    {
        "interesse": "praias"
    }
)
print(response)
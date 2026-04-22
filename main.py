from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

prompt_cidade = PromptTemplate(
    template="""
    Sugira uma cidade dado o meu interesse por {interesse}.
    """,
    input_variables=["interesse"],
)

modelo = ChatOpenAI(
    model="gpt-3.5-turbo", 
    temperature=0.9, 
    api_key=api_key
)

chain = prompt_cidade | modelo | StrOutputParser()


response = chain.invoke(
    {
        "interesse": "praias"
    }
)
print(response)
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default="python")
parser.add_argument("--task", type=str, default="return a list of 10 numbers")
args = parser.parse_args()

load_dotenv()

# Initialize the ChatAnthropic model (Opus)
llm = ChatAnthropic(model="claude-3-opus-20240229")

# Define the prompt template
code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}.",
    input_variables=["language", "task"]
)

# Create a RunnableSequence
chain = (
    RunnableParallel(
        language=RunnablePassthrough(),
        task=RunnablePassthrough()
    )
    | code_prompt
    | llm
    | StrOutputParser()
)

code = chain.invoke({"language": args.language, "task": args.task})
print(code)
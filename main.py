from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
import os
import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default="python")
parser.add_argument("--task", type=str, default="return a list of 10 numbers")
args = parser.parse_args()

# Initialize the ChatAnthropic model (Opus)
llm = ChatAnthropic(model="claude-3-opus-20240229")

# Define the prompt template
code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}. Do NOT include any explanations or other text than the function itself.",
    input_variables=["language", "task"]
)

test_prompt = PromptTemplate(
    template="Write one test case for the following {language} function:\n{code}. Provide only the unit test code without any explanations.",
    input_variables=["language", "code"]
)

# Create a sequential chain that generates code and then a test
sequential_chain = (
    RunnableParallel(
        language=RunnablePassthrough(),
        task=RunnablePassthrough(),
        code=code_prompt | llm | StrOutputParser()
    )
    | RunnableParallel(
        language=lambda x: x["language"],
        code=lambda x: x["code"],
        test=test_prompt | llm | StrOutputParser()
    )
)

# Run the combined chain
result = sequential_chain.invoke({
    "language": args.language,
    "task": args.task
})

print("Generated Code:")
print(result["code"])
print("\nGenerated Unit Test:")
print(result["test"])
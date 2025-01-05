from dotenv import load_dotenv
import os
from langchain_openai import OpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import LLMMathChain
from langchain.agents import Tool
from langchain_core.prompts import PromptTemplate

# Load environment variables from the .env file
load_dotenv()

# Get the OPENAI API key from the environment
OPENAI_API_KEY = os.getenv("OPEN_AI_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_KEY not found in the .env file")

# Initialize the OpenAI LLM
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key=OPENAI_API_KEY, temperature=0.5)

# Initialize the LLMMathChain
llm_math = LLMMathChain.from_llm(OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key=OPENAI_API_KEY, temperature=0.5))

# Define a tool for mathematical computations
math_tool = Tool(
    name="Calculator",
    func=llm_math.run,
    description="Useful for when you need to answer questions about math."
)

# Collect all tools into a list
tools = [math_tool]

# Print the name and description of each tool
for tool in tools:
    print(f"Tool Name: {tool.name}\nDescription: {tool.description}\n")

# Define the Prompt Template
template = '''Answer the following questions as best you can. You have access to the following tools:
{tools}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

# Create the Zero-Shot Agent
zero_shot_agent = create_react_agent(
    tools=tools,
    llm=llm,
    prompt=prompt
)

# Initialize the Agent Executor
agent_executor = AgentExecutor(
    agent=zero_shot_agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True
)

# Function to handle multiple math calculations
def handle_multiple_calculations(questions):
    results = []
    for question in questions:
        print(f"\nProcessing Question: {question}")
        response = agent_executor.invoke({"input": question})
        results.append({"question": question, "response": response})
        print(f"Response: {response}\n")
    return results

# Example list of math questions
math_questions = [
    "What is root over 25?",
    "What is 5 multiplied by 8?",
    "What is the square of 12?",
    "What is 100 divided by 4?"
]

# Process the questions
responses = handle_multiple_calculations(math_questions)

# Print all results
print("\nFinal Responses:")
for res in responses:
    print(f"Question: {res['question']}\nResponse: {res['response']}\n")

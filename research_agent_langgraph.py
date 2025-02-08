import os
import requests
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.tools import Tool
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Tavily API Key
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Azure OpenAI GPT-4
llm = AzureChatOpenAI(
    deployment_name="gpt-4o",
    openai_api_base="https://jnkeastusaihub1808394659.openai.azure.com/",
    openai_api_key="65c7jll11Y0tmXi60jK8WC4XIV4Trj1KEZtFZCUlsxLucWxM7p9WJQQJ99BAACYeBjFXJ3w3AAAAACOG3pai",
    openai_api_version="2023-05-15",
    temperature=0.7
)

### Tavily Search Tool ###
def search_tavily(query: str):
    """
    Tool to search Tavily API.
    """
    url = "https://api.tavily.com/search"
    headers = {
        "Authorization": f"Bearer {tavily_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "num_results": 10
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        results = data["results"]
        return "\n\n".join([f"Title: {res['title']}\nURL: {res['url']}" for res in results[:5]])
    except Exception as e:
        return f"Error querying Tavily API: {e}"

tavily_tool = Tool(
    name="Tavily Search",
    func=search_tavily,
    description="Search Tavily for information on any topic."
)

### GPT-4 Tool ###
def gpt_qa_tool(prompt: str):
    """
    Tool to answer a question or perform a task using Azure OpenAI GPT-4.
    """
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        response = llm.predict_messages(messages)
    except Exception as e:
        return f"Error in GPT-4 response: {e}"

gpt_tool = Tool(
    name="GPT-4 Assistant",
    func=gpt_qa_tool,
    description="Answer any question or perform tasks using context."
)

### Initialize the Agent ###
tools = [tavily_tool, gpt_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Fully dynamic agent
    verbose=True  # Enable logging
)

### Interactive Agent ###
def run_dynamic_agent():
    print("Welcome! I am your AI assistant. Ask me anything.")
    print("Type 'exit' to end the session.")
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        # Exit condition
        if user_input.lower() == "exit":
            print("Goodbye! Have a great day!")
            break

        # Dynamic response
        try:
            response = agent.run(user_input)
            print(f"Agent: {response}")
        except openai.error.OpenAIError as e:
            print(f"OpenAI API Error: {e}")
        except Exception as e:
            print(f"General Error: {e}")

### Main Function ###
if __name__ == "__main__":
    run_dynamic_agent()
import asyncio
import streamlit as st

from src.agents.research.agent import ResearchAgent
from src.core.llm import create_llm
from src.config.settings import load_settings
from src.core.tools import create_tavily_tool
from langchain.agents import AgentType


# LLM and Agent setup
settings = load_settings()
llm = create_llm(azure_settings=settings.azure)

# Initialize tools (can be expanded based on needs)
search_tool = create_tavily_tool(api_key=settings.tavily_api_key)
tools = [search_tool]

# Initialize agent with required parameters
agent = ResearchAgent(
    llm=llm,
    tools=tools,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
agent.initialize()

async def run_research_async(company, audience, follow_up=None):
    """Run research with the agent based on company and audience"""
    if follow_up:
        prompt = f"Given the target company {company} and target audience {audience}, {follow_up}"
    else:
        prompt = f"Research market opportunities and strategies for {company} targeting {audience}. Focus on market size, customer needs, and potential strategies."
    
    try:
        response = await agent.run(prompt)
        return response
    except Exception as e:
        return f"Error during research: {str(e)}"

def run_research(company, audience, follow_up=None):
    """Synchronous wrapper for the async research function"""
    return asyncio.run(run_research_async(company, audience, follow_up))

# Set page config
st.set_page_config(page_title="Market Research Assistant", layout="wide")

# Title
st.title("Market Research Assistant")

# Initialize session state for storing research history
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
    st.session_state.current_company = ""
    st.session_state.current_audience = ""

# Input fields
col1, col2 = st.columns(2)
with col1:
    company = st.text_input("Target Company", key="company")
with col2:
    audience = st.text_input("Target Audience", key="audience")

# Go ahead button
if st.button("Go Ahead"):
    if company and audience:
        st.session_state.current_company = company
        st.session_state.current_audience = audience
        
        with st.spinner("Researching..."):
            result = run_research(company, audience)
            st.session_state.research_history.append({
                "type": "initial",
                "result": result
            })
    else:
        st.error("Please enter both company and audience")

# Follow-up research section
if st.session_state.research_history:
    st.subheader("Follow-up Research")
    follow_up = st.text_area("Enter additional research questions or aspects to explore")
    
    if st.button("Do More Research"):
        if follow_up:
            with st.spinner("Conducting follow-up research..."):
                result = run_research(
                    st.session_state.current_company,
                    st.session_state.current_audience,
                    follow_up
                )
                st.session_state.research_history.append({
                    "type": "follow_up",
                    "query": follow_up,
                    "result": result
                })
        else:
            st.error("Please enter a follow-up question")

# Display research history
if st.session_state.research_history:
    st.subheader("Research Results")
    for i, research in enumerate(st.session_state.research_history):
        with st.expander(
            f"{'Initial Research' if research['type'] == 'initial' else 'Follow-up Research'} #{i+1}",
            expanded=(i == len(st.session_state.research_history) - 1)
        ):
            if research['type'] == 'follow_up':
                st.write("**Question:**", research['query'])
            st.write("**Results:**")
            st.write(research['result'])
            st.divider()

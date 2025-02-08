"""
Research agent implementation.
"""
from typing import Dict, List, Optional
from langchain.agents import AgentType
from langchain.chat_models import AzureChatOpenAI
from langchain.tools import Tool
from ..base import BaseAgent
from .prompts import RESEARCH_AGENT_PROMPT, QUESTION_GENERATION_PROMPT, DATA_ANALYSIS_PROMPT

class ResearchAgent(BaseAgent):
    """
    Agent specialized in company research and analysis.
    """
    def __init__(
        self,
        llm: AzureChatOpenAI,
        tools: List[Tool],
        agent_type: AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose: bool = True
    ):
        """
        Initialize the research agent.
        
        Args:
            llm: Language model instance
            tools: List of tools available to the agent
            agent_type: Type of agent to initialize
            verbose: Whether to enable verbose logging
        """
        super().__init__(llm, tools, agent_type, verbose)
        self.research_chain = RESEARCH_AGENT_PROMPT
        self.question_chain = QUESTION_GENERATION_PROMPT
        self.analysis_chain = DATA_ANALYSIS_PROMPT
        self.collected_data: Dict[str, str] = {}
        
    def _post_initialize(self) -> None:
        """
        Additional initialization steps for research agent.
        """
        # Could add custom initialization logic here
        pass
    
    async def generate_questions(self, company_name: str) -> str:
        """
        Generate research questions for the target company.
        
        Args:
            company_name: Name of the company to research
            
        Returns:
            str: Generated research questions
        """
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
            
        response = await self.llm.apredict_messages(
            self.question_chain.format_messages(company_name=company_name)
        )
        return response.content
    
    async def analyze_data(self, collected_data: str) -> str:
        """
        Analyze collected company data.
        
        Args:
            collected_data: Raw collected data about the company
            
        Returns:
            str: Structured analysis of the data
        """
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
            
        response = await self.llm.apredict_messages(
            self.analysis_chain.format_messages(collected_data=collected_data)
        )
        return response.content
    
    async def run(self, input_text: str) -> str:
        """
        Run the research agent with the given input.
        
        Args:
            input_text: Input text describing the research task
            
        Returns:
            str: Agent's research findings
        """
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        try:
            # Generate research questions
            questions = await self.generate_questions(input_text)
            
            # Execute research using agent
            raw_findings = await self.agent.arun(
                self.research_chain.format(input=input_text)
            )
            
            # Store collected data
            self.collected_data[input_text] = raw_findings
            
            # Analyze findings
            analysis = await self.analyze_data(raw_findings)
            
            # Combine results
            final_report = (
                f"Research Questions:\n{questions}\n\n"
                f"Raw Findings:\n{raw_findings}\n\n"
                f"Analysis:\n{analysis}"
            )
            
            return final_report
            
        except Exception as e:
            return f"Error during research: {str(e)}"

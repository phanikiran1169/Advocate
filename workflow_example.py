"""
End-to-end workflow example demonstrating the complete process from research to ad generation.
"""
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from langchain.agents import AgentType
from langchain.tools import Tool

from src.agents.research.agent import ResearchAgent
from src.agents.marketing.agent import MarketingAgent
from src.agents.marketing.campaign_generator import CampaignIdeaGenerator
from src.agents.AdGen.test import process_campaigns, save_processed_campaigns
from src.core.claude_llm import create_claude_llm
from src.config.settings import load_settings

# Load settings
settings = load_settings()

async def run_research_phase(company_name: str) -> str:
    """
    Phase 1: Company Research
    Uses ResearchAgent to gather and analyze company information.
    """
    print("\n=== Phase 1: Research ===")
    print(f"Researching company: {company_name}")
    
    # Initialize research agent
    llm = create_claude_llm(api_key=settings.claude_api_key)
    research_agent = ResearchAgent(
        llm=llm,
        tools=[],  # Add any specific research tools here
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    # Initialize the agent
    await research_agent.initialize()
    
    # Run research
    research_report = await research_agent.run(company_name)
    
    print("\nResearch Phase Output:")
    print("-" * 50)
    print(research_report)
    print("-" * 50)
    
    return research_report

async def run_marketing_phase(research_report: str) -> List[Dict]:
    """
    Phase 2: Marketing Analysis and Campaign Generation
    Uses MarketingAgent to analyze research and generate campaign ideas.
    """
    print("\n=== Phase 2: Marketing ===")
    print("Generating marketing campaigns based on research...")
    
    # Initialize marketing agent
    llm = create_claude_llm(api_key=settings.claude_api_key)
    marketing_agent = MarketingAgent(
        llm=llm,
        tools=[],  # Add any specific marketing tools here
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    # Initialize the agent
    await marketing_agent.initialize()
    
    # Generate campaigns
    campaign_ideas = await marketing_agent.run(research_report)
    
    print("\nMarketing Phase Output:")
    print("-" * 50)
    print(json.dumps(campaign_ideas, indent=2))
    print("-" * 50)
    
    return campaign_ideas

def run_ad_generation_phase(campaigns: List[Dict]) -> str:
    """
    Phase 3: Advertisement Generation
    Uses AdGen to create final advertisement content for each campaign.
    """
    print("\n=== Phase 3: Ad Generation ===")
    print("Generating advertisement content for campaigns...")
    
    # Process campaigns through ad generator
    processed_campaigns = process_campaigns(campaigns)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"generated_campaigns_{timestamp}.json"
    save_processed_campaigns(processed_campaigns, output_file)
    
    print("\nAd Generation Phase Output:")
    print("-" * 50)
    print(json.dumps(processed_campaigns, indent=2))
    print("-" * 50)
    
    return output_file

async def main():
    """
    Run the complete workflow with a sample company.
    """
    print("Starting End-to-End Marketing Workflow Example")
    print("=" * 50)
    
    # Sample company information
    company_name = "EcoTech Solutions"
    print(f"\nExample Company: {company_name}")
    print("A sustainable technology company specializing in smart home devices")
    
    try:
        # Phase 1: Research
        research_report = await run_research_phase(company_name)
        
        # Phase 2: Marketing
        campaign_ideas = await run_marketing_phase(research_report)
        
        # Phase 3: Ad Generation
        output_file = run_ad_generation_phase(campaign_ideas)
        
        print("\nWorkflow Completed Successfully!")
        print(f"Final output saved to: {output_file}")
        
    except Exception as e:
        print(f"\nError in workflow: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

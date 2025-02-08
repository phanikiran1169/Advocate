import asyncio
from typing import Dict, List
import logging
from datetime import datetime

from src.agents.marketing.campaign_generator import CampaignIdeaGenerator
from src.agents.AdGen.ad_content_generator import CreativeAgent
from src.agents.AdGen.orchestrator import AdCampaignOrchestrator
from src.core.claude_llm import create_claude_llm
from src.config.settings import load_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load settings
settings = load_settings()

async def generate_marketing_strategy(company_name: str, target_audience: str) -> List[Dict]:
    """
    Generate marketing campaign ideas based on company and audience.
    
    Args:
        company_name: Name of the company
        target_audience: Description of the target audience
        
    Returns:
        List[Dict]: List of campaign ideas
    """
    logger.info(f"Generating marketing strategy for {company_name}")
    
    # Create company analysis structure expected by CampaignIdeaGenerator
    company_analysis = {
        "company_summary": f"""
        {company_name} is looking to create impactful marketing campaigns
        that will resonate with their target audience and achieve their
        marketing objectives.
        """,
        "target_audience": target_audience,
        "brand_values": """
        Innovation, Quality, Customer Focus, Market Leadership, Professional Excellence
        """
    }
    
    # Initialize campaign generator
    campaign_generator = CampaignIdeaGenerator(num_campaigns=3)
    
    # Generate campaign ideas
    campaign_ideas = await campaign_generator.generate_campaign_ideas(company_analysis)
    return campaign_ideas

async def generate_ad_assets(campaign_ideas: List[Dict]) -> List[Dict]:
    """
    Generate complete ad assets for each campaign idea.
    
    Args:
        campaign_ideas: List of campaign ideas to process
        
    Returns:
        List[Dict]: Generated campaign assets and their locations
    """
    logger.info("Generating ad assets for campaigns")
    
    # Initialize creative agent with Claude LLM
    llm = create_claude_llm(api_key=settings.claude_api_key)
    creative_agent = CreativeAgent(llm=llm, tools=[], verbose=True)
    await creative_agent.initialize()
    
    # Initialize orchestrator
    orchestrator = AdCampaignOrchestrator(creative_agent=creative_agent)
    
    # Generate assets for all campaigns
    results = await orchestrator.generate_campaign_assets({"company_analysis": campaign_ideas})
    
    return results

async def main(company_name: str, target_audience: str):
    """
    Run the complete ad generation workflow.
    
    Args:
        company_name: Name of the company
        target_audience: Description of the target audience
    """
    try:
        # Step 1: Generate marketing strategy and campaign ideas
        campaign_ideas = await generate_marketing_strategy(company_name, target_audience)
        logger.info(f"Generated {len(campaign_ideas)} campaign ideas")
        
        # Step 2: Generate ad assets for each campaign
        results = await generate_ad_assets(campaign_ideas)
        logger.info(f"Generated assets for {len(results)} campaigns")
        
        # Print results summary
        print("\nGenerated Campaign Assets:")
        print("-" * 50)
        for result in results:
            print(f"\nCampaign: {result['campaign_name']}")
            print(f"Directory: {result['campaign_dir']}")
            print("\nAssets:")
            for asset_type, path in result['assets'].items():
                print(f"- {asset_type}: {path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in workflow: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    company = "EcoTech Solutions"
    audience = "Environmentally conscious homeowners aged 30-50 interested in smart home technology"
    
    asyncio.run(main(company, audience))

import asyncio
import os
from langchain.chat_models import AzureChatOpenAI
from dotenv import load_dotenv
from .ad_content_generator import CreativeAgent
from .orchestrator import AdCampaignOrchestrator

async def test_ad_generation():
    # Load environment variables
    load_dotenv()
    
    # Initialize Azure OpenAI
    llm = AzureChatOpenAI(
        deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
        openai_api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
        azure_endpoint=os.getenv('AZURE_OPENAI_API_BASE'),
        api_key=os.getenv('AZURE_OPENAI_API_KEY')
    )
    
    # Initialize Creative Agent
    creative_agent = CreativeAgent(llm=llm, tools=[])
    
    # Initialize Orchestrator
    orchestrator = AdCampaignOrchestrator(creative_agent)
    
    # Test company analysis
    test_analysis = {
        "company_summary": """
        EcoTech Solutions is a sustainable technology company specializing in smart home devices.
        They produce energy-efficient thermostats and power monitoring systems that help homeowners
        reduce their carbon footprint while saving money. Their products feature sleek, modern design
        and integrate with most smart home ecosystems.
        """,
        "target_audience": """
        Environmentally conscious homeowners aged 25-45, tech-savvy professionals who value both
        sustainability and modern design. They are willing to invest in quality products that help
        reduce their environmental impact while maintaining a comfortable lifestyle.
        """,
        "brand_values": """
        Innovation, Sustainability, User-Friendly Design, Environmental Responsibility, Quality
        """
    }
    
    try:
        # Generate campaign assets
        results = await orchestrator.generate_campaign_assets(test_analysis)
        
        # Print results
        print("\nGenerated Campaigns:")
        for result in results:
            print(f"\nCampaign: {result['campaign_name']}")
            print(f"Directory: {result['campaign_dir']}")
            print("Assets:")
            for asset_type, path in result['assets'].items():
                print(f"- {asset_type}: {path}")
                
    except Exception as e:
        print(f"Error in test: {str(e)}")

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_ad_generation())

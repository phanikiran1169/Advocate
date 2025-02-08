"""
Marketing agent implementation for advertisement generation.
"""
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from langchain.agents import AgentType
from langchain.chat_models import AzureChatOpenAI
from langchain.tools import Tool
from langchain_anthropic import ChatAnthropic
from ..base import BaseAgent
from .campaign_generator import CampaignIdeaGenerator
from .prompts import (
    MARKETING_AGENT_PROMPT,
    BRAND_ANALYSIS_PROMPT,
    AUDIENCE_MAPPING_PROMPT,
    MARKET_POSITION_PROMPT,
    AD_GENERATION_PROMPT
)
from models.vectorstore.base import BaseVectorStore

class MarketingAgent(BaseAgent):
    """
    Agent specialized in generating marketing and advertisement content from research data.
    """
    def __init__(
        self,
        llm: AzureChatOpenAI,
        tools: List[Tool],
        agent_type: AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose: bool = True,
        vectorstore: Optional[BaseVectorStore] = None,
        num_campaigns: int = 5
    ):
        """
        Initialize the marketing agent.
        
        Args:
            llm: Language model instance
            tools: List of tools available to the agent
            agent_type: Type of agent to initialize
            verbose: Whether to enable verbose logging
            vectorstore: Optional vector store for storing generated content
            num_campaigns: Number of campaign ideas to generate
        """
        super().__init__(llm, tools, agent_type, verbose)
        self.marketing_chain = MARKETING_AGENT_PROMPT
        self.brand_chain = BRAND_ANALYSIS_PROMPT
        self.audience_chain = AUDIENCE_MAPPING_PROMPT
        self.market_chain = MARKET_POSITION_PROMPT
        self.ad_chain = AD_GENERATION_PROMPT
        self.generated_content: Dict[str, Dict[str, str]] = {}
        self.vectorstore = vectorstore
        self.session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.campaign_generator = CampaignIdeaGenerator(num_campaigns=num_campaigns)
        
    def _post_initialize(self) -> None:
        """
        Additional initialization steps for marketing agent.
        """
        # Could add custom initialization logic here
        pass
    
    async def analyze_brand(self, research_data: str) -> str:
        """
        Analyze brand voice and personality from research data.
        
        Args:
            research_data: Research findings about the company
            
        Returns:
            str: Brand voice analysis
        """
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
            
        response = await self.llm.apredict_messages(
            self.brand_chain.format_messages(research_data=research_data)
        )
        return response.content
    
    async def map_audience(self, research_data: str) -> str:
        """
        Create detailed target audience profiles from research data.
        
        Args:
            research_data: Research findings about the company
            
        Returns:
            str: Target audience profiles
        """
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
            
        response = await self.llm.apredict_messages(
            self.audience_chain.format_messages(research_data=research_data)
        )
        return response.content
    
    async def assess_market_position(self, research_data: str) -> str:
        """
        Analyze market position and competitive advantages.
        
        Args:
            research_data: Research findings about the company
            
        Returns:
            str: Market position analysis
        """
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
            
        response = await self.llm.apredict_messages(
            self.market_chain.format_messages(research_data=research_data)
        )
        return response.content
    
    async def generate_campaign_ideas(self, brand_analysis: str, audience_profiles: str, market_analysis: str) -> List[Dict]:
        """
        Generate campaign ideas using the CampaignIdeaGenerator.
        
        Args:
            brand_analysis: Analysis of brand voice and personality
            audience_profiles: Target audience analysis
            market_analysis: Market position analysis
            
        Returns:
            List[Dict]: List of campaign ideas with details
        """
        company_analysis = {
            "company_summary": market_analysis,
            "target_audience": audience_profiles,
            "brand_values": brand_analysis
        }
        
        return await self.campaign_generator.generate_campaign_ideas(company_analysis)

    async def generate_ad_content(self, campaigns: List[Dict]) -> Tuple[List[Dict], str]:
        """
        Generate detailed advertisement content for each campaign using the AdGen system.
        
        Args:
            campaigns: List of campaign dictionaries
            
        Returns:
            Tuple[List[Dict], str]: Processed campaigns with generated content and output file path
        """
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")

        try:
            # Import the ad generation functions
            from ..AdGen.ad_processor import process_campaigns, save_processed_campaigns
            
            # Process campaigns through ad generator
            processed_campaigns = process_campaigns(campaigns)
            
            # Save processed campaigns
            output_file = f"generated_campaigns_{self.session_id}.json"
            output_path = str(Path("src/agents/AdGen") / output_file)
            save_processed_campaigns(processed_campaigns, output_path)
            
            return processed_campaigns, output_path
            
        except Exception as e:
            logger.error(f"Error generating ad content: {str(e)}")
            raise RuntimeError(f"Ad content generation failed: {str(e)}")
    
    async def run(self, research_report: str) -> str:
        """
        Run the marketing agent to generate advertisement content.
        
        Args:
            research_report: Research report to analyze
            
        Returns:
            str: Generated marketing and advertisement content
        """
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        try:
            # Analyze brand elements
            brand_analysis = await self.analyze_brand(research_report)
            if self.vectorstore:
                self.vectorstore.add_texts(
                    texts=[brand_analysis],
                    metadatas=[{
                        "content_type": "brand_analysis",
                        "analysis_type": "voice_and_personality"
                    }],
                    session_id=self.session_id
                )
            
            # Map target audience
            audience_profiles = await self.map_audience(research_report)
            if self.vectorstore:
                self.vectorstore.add_texts(
                    texts=[audience_profiles],
                    metadatas=[{
                        "content_type": "audience_analysis",
                        "analysis_type": "profiles_and_segments"
                    }],
                    session_id=self.session_id
                )
            
            # Assess market position
            market_analysis = await self.assess_market_position(research_report)
            if self.vectorstore:
                self.vectorstore.add_texts(
                    texts=[market_analysis],
                    metadatas=[{
                        "content_type": "market_analysis",
                        "analysis_type": "position_and_competition"
                    }],
                    session_id=self.session_id
                )
            
            # Combine analyses
            combined_analysis = (
                f"Brand Analysis:\n{brand_analysis}\n\n"
                f"Target Audience:\n{audience_profiles}\n\n"
                f"Market Position:\n{market_analysis}"
            )
            
            # Generate campaign ideas
            campaign_ideas = await self.generate_campaign_ideas(
                brand_analysis,
                audience_profiles,
                market_analysis
            )
            
            if self.vectorstore:
                self.vectorstore.add_texts(
                    texts=[str(campaign_ideas)],
                    metadatas=[{
                        "content_type": "campaign_ideas",
                        "analysis_type": "creative_concepts"
                    }],
                    session_id=self.session_id
                )
            
            # Generate detailed advertisement content
            processed_campaigns, output_path = await self.generate_ad_content(campaign_ideas)
            
            if self.vectorstore:
                self.vectorstore.add_texts(
                    texts=[str(processed_campaigns)],
                    metadatas=[{
                        "content_type": "ad_content",
                        "analysis_type": "generated_advertisements",
                        "output_file": output_path
                    }],
                    session_id=self.session_id
                )
            
            # Store generated content
            self.generated_content[self.session_id] = {
                "brand_analysis": brand_analysis,
                "audience_profiles": audience_profiles,
                "market_analysis": market_analysis,
                "campaign_ideas": campaign_ideas,
                "processed_campaigns": processed_campaigns,
                "output_file": output_path
            }
            
            # Validate campaign ideas
            if not campaign_ideas or not isinstance(campaign_ideas, list):
                logger.error("Failed to generate valid campaign ideas")
                raise RuntimeError("No valid campaign ideas were generated")
                
            if len(campaign_ideas) == 0:
                logger.error("Generated empty campaign ideas list")
                raise RuntimeError("Generated campaign ideas list is empty")
                
            # Validate each campaign has required fields
            required_fields = ['campaign_name', 'core_message', 'visual_theme_description']
            for campaign in campaign_ideas:
                missing_fields = [field for field in required_fields if not campaign.get(field)]
                if missing_fields:
                    logger.error(f"Campaign missing required fields: {missing_fields}")
                    raise RuntimeError(f"Campaign data incomplete. Missing: {', '.join(missing_fields)}")
            
            # Store in vectorstore if available
            if self.vectorstore:
                self.vectorstore.add_texts(
                    texts=[str(campaign_ideas)],
                    metadatas=[{
                        "content_type": "marketing_analysis",
                        "analysis_type": "campaign_ideas",
                        "timestamp": datetime.utcnow().isoformat()
                    }],
                    session_id=self.session_id
                )
            
            return campaign_ideas
            
        except Exception as e:
            logger.error(f"Error during marketing analysis: {str(e)}")
            raise RuntimeError(f"Marketing analysis failed: {str(e)}")

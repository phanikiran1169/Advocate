import os
import json
import asyncio
from typing import Dict, List
import logging
from datetime import datetime
from .ad_content_generator import CreativeAgent
from .image_gen import SDXLTurboGenerator
from ..marketing.campaign_generator import CampaignIdeaGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdCampaignOrchestrator:
    """
    Orchestrates the complete ad campaign generation workflow, from campaign ideas to final assets.
    """
    def __init__(self, creative_agent: CreativeAgent):
        """
        Initialize the orchestrator.
        
        Args:
            creative_agent: Initialized CreativeAgent instance
        """
        self.creative_agent = creative_agent
        self.campaign_generator = CampaignIdeaGenerator()
        self.image_generator = SDXLTurboGenerator()
        self.output_dir = "Outputs"

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize the filename to be safe for all operating systems."""
        # Replace spaces and special characters
        sanitized = "".join(c if c.isalnum() else "_" for c in filename)
        return sanitized.strip("_")

    def _create_campaign_directory(self, campaign_name: str) -> str:
        """Create a directory for the campaign assets."""
        sanitized_name = self._sanitize_filename(campaign_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        campaign_dir = os.path.join(self.output_dir, f"{sanitized_name}_{timestamp}")
        
        os.makedirs(campaign_dir, exist_ok=True)
        return campaign_dir

    def _save_text_asset(self, campaign_dir: str, filename: str, content: str):
        """Save a text asset to the campaign directory."""
        file_path = os.path.join(campaign_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path

    async def generate_campaign_assets(self, company_analysis: Dict) -> List[Dict]:
        """
        Generate complete ad campaigns including all assets.
        
        Args:
            company_analysis: Dictionary containing company analysis
            
        Returns:
            List[Dict]: List of generated campaigns with asset paths
        """
        try:
            # Create main output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Generate campaign ideas
            campaigns = await self.campaign_generator.generate_campaign_ideas(company_analysis)
            
            # Process each campaign
            results = []
            for campaign in campaigns:
                try:
                    # Create campaign directory
                    campaign_dir = self._create_campaign_directory(campaign['campaign_name'])
                    
                    # Generate creative assets
                    assets = await self.creative_agent.generate_campaign_assets(campaign)
                    
                    # Save tagline
                    tagline_path = self._save_text_asset(
                        campaign_dir, 
                        'tagline.txt',
                        assets['tagline']
                    )
                    
                    # Save story
                    story_path = self._save_text_asset(
                        campaign_dir,
                        'story.txt',
                        assets['story']
                    )
                    
                    # Generate and save image
                    image_path = self.image_generator.generate_image(
                        assets['image_prompt'],
                        output_dir=campaign_dir
                    )
                    
                    # Save campaign details
                    campaign_details = {
                        **campaign,
                        'generated_assets': {
                            'tagline_path': tagline_path,
                            'story_path': story_path,
                            'image_path': image_path,
                            'tagline': assets['tagline'],
                            'story': assets['story'],
                            'image_prompt': assets['image_prompt']
                        }
                    }
                    
                    details_path = self._save_text_asset(
                        campaign_dir,
                        'campaign_details.json',
                        json.dumps(campaign_details, indent=2)
                    )
                    
                    results.append({
                        'campaign_name': campaign['campaign_name'],
                        'campaign_dir': campaign_dir,
                        'assets': {
                            'tagline': tagline_path,
                            'story': story_path,
                            'image': image_path,
                            'details': details_path
                        }
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing campaign '{campaign['campaign_name']}': {str(e)}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error in campaign generation workflow: {str(e)}")
            raise

    async def generate_single_campaign(self, campaign: Dict) -> Dict:
        """
        Generate assets for a single campaign.
        
        Args:
            campaign: Dictionary containing campaign details
            
        Returns:
            Dict: Generated campaign with asset paths
        """
        try:
            # Create campaign directory
            campaign_dir = self._create_campaign_directory(campaign['campaign_name'])
            
            # Generate creative assets
            assets = await self.creative_agent.generate_campaign_assets(campaign)
            
            # Save tagline
            tagline_path = self._save_text_asset(
                campaign_dir,
                'tagline.txt',
                assets['tagline']
            )
            
            # Save story
            story_path = self._save_text_asset(
                campaign_dir,
                'story.txt',
                assets['story']
            )
            
            # Generate and save image
            image_path = self.image_generator.generate_image(
                assets['image_prompt'],
                output_dir=campaign_dir
            )
            
            # Save campaign details
            campaign_details = {
                **campaign,
                'generated_assets': {
                    'tagline_path': tagline_path,
                    'story_path': story_path,
                    'image_path': image_path,
                    'tagline': assets['tagline'],
                    'story': assets['story'],
                    'image_prompt': assets['image_prompt']
                }
            }
            
            details_path = self._save_text_asset(
                campaign_dir,
                'campaign_details.json',
                json.dumps(campaign_details, indent=2)
            )
            
            return {
                'campaign_name': campaign['campaign_name'],
                'campaign_dir': campaign_dir,
                'assets': {
                    'tagline': tagline_path,
                    'story': story_path,
                    'image': image_path,
                    'details': details_path
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating campaign assets: {str(e)}")
            raise

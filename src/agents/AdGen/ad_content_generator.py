from typing import Dict, List, Optional
from langchain.agents import AgentType
from langchain.chat_models import AzureChatOpenAI
from langchain.tools import Tool
from ..base import BaseAgent
from .prompts import TAGLINE_PROMPT, NARRATIVE_PROMPT, IMAGE_PROMPT


class CreativeAgent(BaseAgent):
    """
    Agent specialized in generating taglines, narratives, and image prompts based on input data.
    """
    def __init__(
        self,
        llm: AzureChatOpenAI,
        tools: List[Tool],
        agent_type: AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose: bool = True
    ):
        """
        Initialize the creative agent.
        
        Args:
            llm: Language model instance
            tools: List of tools available to the agent
            agent_type: Type of agent to initialize
            verbose: Whether to enable verbose logging
        """
        super().__init__(llm, tools, agent_type, verbose)
        self.tagline_chain = TAGLINE_PROMPT
        self.narrative_chain = NARRATIVE_PROMPT
        self.image_chain = IMAGE_PROMPT
        self.data: Optional[pd.DataFrame] = None

    def _post_initialize(self) -> None:
        """
        Additional initialization steps for creative agent.
        """
        # Could add custom initialization logic here
        pass

    def load_database(self, file_path: str) -> None:
        """
        Load a database file into a DataFrame.
        
        Args:
            file_path: Path to the database file (CSV, JSON, etc.)
        """
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                self.data = pd.read_json(file_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or JSON.")
        except Exception as e:
            raise RuntimeError(f"Error loading database: {str(e)}")

    async def generate_tagline(self, core_message: str, visual_theme: str, emotional_appeal: str) -> str:
        """
        Generate a tagline based on campaign elements.
        
        Args:
            core_message: The main message of the campaign
            visual_theme: Description of the visual theme
            emotional_appeal: Intended emotional impact
            
        Returns:
            str: Generated tagline
        """
        response = await self.llm.apredict_messages(
            self.tagline_chain.format_messages(
                core_message=core_message,
                visual_theme=visual_theme,
                emotional_appeal=emotional_appeal
            )
        )
        return response.content

    async def generate_story(self, core_message: str, visual_theme: str, emotional_appeal: str) -> str:
        """
        Generate a narrative based on campaign elements.
        
        Args:
            core_message: The main message of the campaign
            visual_theme: Description of the visual theme
            emotional_appeal: Intended emotional impact
            
        Returns:
            str: Generated narrative
        """
        response = await self.llm.apredict_messages(
            self.narrative_chain.format_messages(
                core_message=core_message,
                visual_theme=visual_theme,
                emotional_appeal=emotional_appeal
            )
        )
        return response.content

    async def generate_image_prompt(self, campaign_name: str, product_prompt: str, brand_prompt: str, social_prompt: str) -> str:
        """
        Generate an image prompt based on campaign elements.
        
        Args:
            campaign_name: Name of the campaign
            product_prompt: Product-focused prompt elements
            brand_prompt: Brand-focused prompt elements
            social_prompt: Social media considerations
            
        Returns:
            str: Generated image prompt
        """
        response = await self.llm.apredict_messages(
            self.image_chain.format_messages(
                campaign_name=campaign_name,
                product_prompt=product_prompt,
                brand_prompt=brand_prompt,
                social_prompt=social_prompt
            )
        )
        return response.content

    async def generate_campaign_assets(self, campaign: Dict) -> Dict[str, str]:
        """
        Generate all creative assets for a campaign.
        
        Args:
            campaign: Dictionary containing campaign details
            
        Returns:
            Dict[str, str]: Dictionary containing generated assets
        """
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        try:
            # Extract campaign elements
            core_message = campaign.get('core_message', '')
            visual_theme = campaign.get('visual_theme_description', '')
            emotional_appeal = campaign.get('key_emotional_appeal', '')
            campaign_name = campaign.get('campaign_name', '')
            prompt_suggestions = campaign.get('prompt_suggestions', {})
            
            # Generate assets
            tagline = await self.generate_tagline(core_message, visual_theme, emotional_appeal)
            story = await self.generate_story(core_message, visual_theme, emotional_appeal)
            image_prompt = await self.generate_image_prompt(
                campaign_name,
                prompt_suggestions.get('product_focused', ''),
                prompt_suggestions.get('brand_focused', ''),
                prompt_suggestions.get('social_media', '')
            )
            
            return {
                'tagline': tagline,
                'story': story,
                'image_prompt': image_prompt
            }
            
        except Exception as e:
            raise RuntimeError(f"Error generating campaign assets: {str(e)}")

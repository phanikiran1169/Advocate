from langchain.prompts import PromptTemplate

TAGLINE_PROMPT = PromptTemplate(
    input_variables=["core_message", "visual_theme", "emotional_appeal"],
    template="""Create a memorable and impactful tagline for an advertisement campaign based on the following elements:

Core Message:
{core_message}

Visual Theme:
{visual_theme}

Emotional Appeal:
{emotional_appeal}

The tagline should be:
1. Concise and memorable (ideally 3-7 words)
2. Capture the essence of the core message
3. Evoke the desired emotional response
4. Align with the visual theme
5. Be distinctive and unique

Generate a single, powerful tagline that meets these criteria."""
)

NARRATIVE_PROMPT = PromptTemplate(
    input_variables=["core_message", "visual_theme", "emotional_appeal"],
    template="""Create a compelling narrative for an advertisement campaign based on the following elements:

Core Message:
{core_message}

Visual Theme:
{visual_theme}

Emotional Appeal:
{emotional_appeal}

The narrative should:
1. Tell a story that resonates with the target audience
2. Incorporate the core message naturally
3. Create vivid imagery that aligns with the visual theme
4. Evoke the intended emotional response
5. Be concise yet impactful (150-200 words)

Generate a narrative that weaves these elements together into a cohesive story."""
)

IMAGE_PROMPT = PromptTemplate(
    input_variables=["product_prompt", "brand_prompt", "social_prompt", "campaign_name"],
    template="""Create a detailed image generation prompt for an advertisement campaign titled "{campaign_name}" based on the following elements:

Product Focus:
{product_prompt}

Brand Elements:
{brand_prompt}

Social Media Considerations:
{social_prompt}

The final image should:
1. Be visually striking and professional
2. Clearly communicate the intended message
3. Follow advertising best practices
4. Be suitable for the target platforms
5. Maintain brand consistency

Generate a detailed prompt that will produce an image meeting these criteria."""
)

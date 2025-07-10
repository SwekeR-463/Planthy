import gradio as gr
from PIL import Image
import os
from dotenv import load_dotenv
from llama_index.multi_modal_llms.gemini import GeminiMultiModal
from llama_index.core.program import MultiModalLLMCompletionProgram
from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.tools import FunctionTool
from llama_index.core import SimpleDirectoryReader
from duckduckgo_search import DDGS
from llama_index.core.agent import ReActAgent
from llama_index.llms.gemini import Gemini


load_dotenv()

token = os.getenv("GEMINI_API_KEY")

class PlantHealth(BaseModel):
    plant_type: str
    condition: str
    symptoms: str
    confidence: float

prompt_template_str = """Analyze the plant image and return:
- Plant type (e.g., tomato, rose)
- Condition (e.g., healthy, diseased)
- Symptoms (e.g., yellow spots, wilting)
- Confidence score (0.0 to 1.0)
"""

# function to analyze leaf
def analyze_plant_image(image_path: str) -> dict:
    gemini_llm = GeminiMultiModal(model_name="gemini-2.0-flash")
    llm_program = MultiModalLLMCompletionProgram.from_defaults(
        output_parser=PydanticOutputParser(PlantHealth),
        image_documents=[SimpleDirectoryReader(input_files=[image_path]).load_data()[0]],
        prompt_template_str=prompt_template_str,
        multi_modal_llm=gemini_llm,
    )
    return llm_program().dict()

# tool for image analysis
image_tool = FunctionTool.from_defaults(
    fn=analyze_plant_image,
    name="analyze_plant_image",
    description="Analyzes a plant image to identify type, condition, and symptoms."
)

# function to search plant info
def search_plant_info(query: str) -> str:
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
        return "\n".join([f"{r['title']}: {r['body']}" for r in results])

# tool for search
search_tool = FunctionTool.from_defaults(
    fn=search_plant_info,
    name="search_plant_info",
    description="Searches the web for plant health and care information."
)

# ReAct agent 
agent = ReActAgent.from_tools(
    tools=[image_tool, search_tool],
    llm=Gemini(model_name="gemini-2.0-flash"),
    verbose=True
)

# function for the flow of gradio app
def plant_health_app(image, user_query):
    image_path = "temp_image.jpg"
    image.save(image_path)
    
    # Query the ReAct agent
    response = agent.chat(f"Analyze this plant image : {image_path}, give a recommendations to cure the issue and return the answer in points. User query: {user_query}")
    
    # Parse response
    recommendations = str(response)
    
    output_text = recommendations
    return output_text

iface = gr.Interface(
    fn=plant_health_app,
    inputs=[
        gr.Image(type="pil", label="Upload Plant Image"),
        gr.Textbox(label="Describe Symptoms or Ask a Question", placeholder="E.g., My tomato plant has yellow spots")
    ],
    outputs=gr.Markdown(label="Diagnosis and Recommendations"),
    title="🌿 Planthy -- Monitor Your Plant Health",
    description="Upload a clear plant image and describe symptoms to get a diagnosis and care tips.",
    theme="huggingface",
    examples=[
        ["tomato.jpeg", "What’s wrong with my tomato plant?"],
        ["rose.jpeg", "Is my rose plant healthy?"]
    ]
)

if __name__ == "__main__":
    iface.launch()
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import os

_ = load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=api_key,
)

async def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"The weather in {city} is 73 degrees and Sunny."

agent = AssistantAgent(
    name="weather_agent",
    model_client=model_client,
    tools=[get_weather],
    system_message="You are a helpful assistant.",
    reflect_on_tool_use=True,
    model_client_stream=True,  # Enable streaming tokens from the model client.
)

async def main() -> None:
    await Console(agent.run_stream(task="What is the weather in New York?"))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
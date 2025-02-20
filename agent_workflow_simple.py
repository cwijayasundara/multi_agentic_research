from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os
from tavily import AsyncTavilyClient
from llama_index.core.agent.workflow import AgentWorkflow
import asyncio
from llama_index.core.workflow import Context

_ = load_dotenv()

api_key=os.getenv("OPENAI_API_KEY")
tvly_api_key = os.getenv("TAVILY_API_KEY")
                  
llm = OpenAI(model="gpt-4o-mini",api_key=api_key)

async def search_web(query: str) -> str:
    """Useful for using the web to answer questions."""
    client = AsyncTavilyClient(api_key=tvly_api_key)
    return str(await client.search(query))

workflow = AgentWorkflow.from_tools_or_functions(
    [search_web],
    llm=llm,
    system_prompt="You are a helpful assistant that can search the web for information.",
)

ctx = Context(workflow)

async def main():
    response = await workflow.run(user_msg="What is the weather in London UK?", ctx=ctx)
    print(str(response))

asyncio.run(main())
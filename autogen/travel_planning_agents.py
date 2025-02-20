from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

_ = load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

planner_llm = OpenAIChatCompletionClient(model="o3-mini", api_key=api_key, temperature=1.0)
exec_llm = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=api_key, temperature=0.2)

planner_agent = AssistantAgent(
    "planner_agent",
    model_client=planner_llm,
    description="A helpful assistant that can plan trips.",
    system_message="You are a helpful assistant that can suggest a travel plan for a user based on their request.",
)

local_agent = AssistantAgent(
    "local_agent",
    model_client=exec_llm,
    description="A local assistant that can suggest local activities or places to visit.",
    system_message="You are a helpful assistant that can suggest authentic and interesting local activities or places to visit for a user and can utilize any context information provided.",
)

language_agent = AssistantAgent(
    "language_agent",
    model_client=exec_llm,
    description="A helpful assistant that can provide language tips for a given destination.",
    system_message="You are a helpful assistant that can review travel plans, providing feedback on important/critical tips about how best to address language or communication challenges for the given destination. If the plan already includes language tips, you can mention that the plan is satisfactory, with rationale.",
)

travel_summary_agent = AssistantAgent(
    "travel_summary_agent",
    model_client=exec_llm,
    description="A helpful assistant that can summarize the travel plan.",
    system_message="You are a helpful assistant that can take in all of the suggestions and advice from the other agents and provide a detailed final travel plan. You must ensure that the final plan is integrated and complete. YOUR FINAL RESPONSE MUST BE THE COMPLETE PLAN. When the plan is complete and all perspectives are integrated, you can respond with TERMINATE.",
)

termination = TextMentionTermination("TERMINATE")
group_chat = RoundRobinGroupChat(
    [planner_agent, local_agent, language_agent, travel_summary_agent], termination_condition=termination
)

import asyncio

async def main():
    await Console(group_chat.run_stream(task="Plan a 5 day trip to Sri Lanka?"))

if __name__ == '__main__':
    asyncio.run(main())
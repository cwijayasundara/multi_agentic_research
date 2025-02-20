from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import QueryEngineTool, ToolMetadata
import os
from llama_index.core.agent import ReActAgent
from llama_parse import LlamaParse

_ = load_dotenv()

llm = OpenAI(model="gpt-4o-mini")

parser = LlamaParse(
    result_type="markdown",
    use_vendor_multimodal_model=True,
    vendor_multimodal_model_name="gemini-2.0-flash-001",
    invalidate_cache=True,
    parsing_instruction="Parse the insurence claim document without loosing any information and keeping the table format intact.",
)

file_name = './data/pb116349-business-health-select-handbook-1024-pdfa.pdf'

def get_tool(name, full_name):
    if not os.path.exists(f"./data/{name}"):
        # build vector index
        documents = parser.load_data(file_name)
        vector_index = VectorStoreIndex.from_documents(documents)
        vector_index.storage_context.persist(persist_dir=f"./data/{name}")
    else:
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=f"./data/{name}"),
        )
    query_engine = vector_index.as_query_engine(similarity_top_k=3, llm=llm)

    query_engine_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name=name,
            description=(
                "Provides information about AXA health insurance policy document"
                f" {full_name}"
            ),
        ),
    )
    return query_engine_tool

rag_tool = get_tool("axa_policy_doc", file_name)

query_engine_tools = [rag_tool]

agent = ReActAgent.from_tools(
    query_engine_tools, llm=llm, verbose=True, max_iterations=20
)

response = agent.chat("Whats the claim amount for dental treatment?")

print(response)

    



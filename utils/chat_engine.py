import json
import os
import logging
import sys
from typing import Generator

from llama_index.indices.vector_store.retrievers import (
    VectorIndexAutoRetriever,
)
from llama_index.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.retrievers import QueryFusionRetriever, BM25Retriever
from llama_index.chat_engine import CondensePlusContextChatEngine
from llama_index.llms.base import ChatMessage

from utils.utils import init_llamaindex_context, init_llamaindex_indices, load_indices, gcs_fs


# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


CONTEXT_PROMPT_TEMPLATE = """
  The following is a friendly conversation between a user and an AI assistant.
  The assistant is talkative and provides lots of specific details from its context.
  If the assistant does not know the answer to a question, it truthfully says it
  does not know. If more information from the user helps to provide more truthful information,
  the assistant asks more question(s).

  Here are the relevant products for the context:

  {context_str}

  Instruction: Based on the above products, provide a detailed answer for the user question below.
  Answer "don't know" if not present in the product.
  Asks more question(s) if not present and the user question can be more clear.
  """


service_context, storage_context, scheme = init_llamaindex_context(
    collection_name=os.environ["collection_name"],
    openai_api_key=os.environ["openai_api_key"],
    qdrant_api_url=os.environ["qdrant_api_url"],
    qdrant_api_key=os.environ["qdrant_api_key"],
    mongo_uri=os.environ["mongo_uri"],
    config_path=os.environ["config_path"],
    scheme_path=os.environ["scheme_path"],
)
indices = init_llamaindex_indices(storage_context, indices=load_indices(os.environ["state_path"]))

similarity_top_k = int(os.environ.get("similarity_top_k") or 2)
num_queries = int(os.environ.get("num_queries") or 1)
similarity_top_k_before_fusion_multiplier = int(os.environ.get("similarity_top_k_before_fusion_multiplier") or 3)

def get_retriever():
    # input_scheme = scheme["input"]
    # vector_store_info = VectorStoreInfo(
    #     content_info=input_scheme["content_info"],
    #     metadata_info=[
    #         MetadataInfo(
    #             name=metadata_info["name"],
    #             type=metadata_info["type"],
    #             description=metadata_info["description"]
    #         )
    #         for metadata_info in input_scheme["metadata_info"]
    #     ],
    # )
    # vector_retriever = VectorIndexAutoRetriever(
    #     indices["vector_index"], vector_store_info=vector_store_info, similarity_top_k=similarity_top_k * similarity_top_k_before_fusion_multiplier)
    vector_retriever = indices["vector_index"].as_retriever(similarity_top_k=similarity_top_k)
    # bm25_retriever = BM25Retriever.from_defaults(docstore=storage_context.docstore, similarity_top_k=similarity_top_k * similarity_top_k_before_fusion_multiplier)

    # TODO: modify here
    retrievers = [
        vector_retriever,
        # bm25_retriever,
        # indices["keyword_table_index"].as_retriever(similarity_top_k=similarity_top_k * similarity_top_k_before_fusion_multiplier)
    ]  # make sure the keyword table is the last position
    # retriever = QueryFusionRetriever(
    #     retrievers,
    #     similarity_top_k=similarity_top_k,
    #     num_queries=num_queries,  # set this to 1 to disable query generation
    #     mode="reciprocal_rerank",
    #     use_async=False,
    #     verbose=False,
    #     # query_gen_prompt="...",  # we could override the query generation prompt here
    # )
    retriever=vector_retriever
    return retriever, retrievers

retriever, _retrievers = get_retriever()

def convert_json_to_stream_data(response):
    return "data: " + json.dumps(response) + "\n\n"

async def handle_session(question, store_session_path) -> Generator:
    if not gcs_fs.exists(store_session_path):
        chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=retriever,
            verbose=False
        )
    else:
        with gcs_fs.open(store_session_path, 'r') as f_p:
            chat_history = json.load(f_p)
        chat_history = [ChatMessage(**model_dict) for model_dict in chat_history]

        chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=retriever,
            chat_history=chat_history,
            verbose=False,
            system_prompt=None,
            context_prompt=CONTEXT_PROMPT_TEMPLATE,
            condense_prompt=None,
        )

    stream_response = chat_engine.stream_chat(question)

    # for _retriever in _retrievers:
    #     response.source_nodes.extend(_retriever.retrieve(response.response))

    node_sources = [{
        "node": {
            "metadata": node.node.metadata,
            "text": node.node.text
        }
    } for node in stream_response.source_nodes]

    yield convert_json_to_stream_data({
        "response": "",
        "source_nodes": node_sources
    })

    text = ""
    for chunk in stream_response.response_gen:
        text += chunk
        response = {
            "response": text,
            "source_nodes": node_sources
        }
        yield "data: " + json.dumps(response) + "\n\n"

    chat_history = [model.dict() for model in chat_engine.chat_history]
    with gcs_fs.open(store_session_path, 'w') as f_p:
        json.dump(chat_history, f_p, indent=2)

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Optional

from agents import Agent, RunContextWrapper, Runner, function_tool
from pydantic import BaseModel, ConfigDict
from raggify_client import RestAPIClient
from typing_extensions import TypedDict

from .logger import logger

__all__ = ["AgentExecutionError", "RagAgentManager"]

# Fixed configuration
_TOPK = 10


class AgentExecutionError(RuntimeError):
    """Wrapper exception raised during openai-agents execution."""


class _TextSearchArgs(TypedDict, total=False):
    query: str


class _RagAgentContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: RestAPIClient
    upload_id: Optional[str] = None


def _format_documents(payload: dict[str, Any]) -> dict[str, Any]:
    """Format search documents into a plain dictionary response.

    Args:
        payload (dict[str, Any]): Response payload returned from the search API.

    Returns:
        dict[str, Any]: Summary dictionary per document.
    """
    docs = payload.get("documents") or []
    if not docs:
        return {"1": "No documents were retrieved."}

    summary_dict = {}
    for idx, doc in enumerate(docs):
        sub_dict = {}
        meta = doc.get("metadata") or {}
        sub_dict["source"] = (
            meta.get("url") or meta.get("file_path") or meta.get("url") or "unknown"
        )
        sub_dict["text"] = doc.get("text", "").strip().replace("\n", " ")
        sub_dict["score"] = doc.get("score", "")
        summary_dict[idx] = sub_dict

    return summary_dict


def _format_response(type: str, query: str, topk: int, payload: dict[str, Any]) -> str:
    """Summarize search results as a JSON string.

    Args:
        type (str): Title describing the result type.
        query (str): Search query or reference file path.
        topk (int): Number of retrieved documents.
        payload (dict[str, Any]): Response payload returned from the search API.

    Returns:
        str: JSON string that summarizes the search results.
    """
    summary = _format_documents(payload)
    result = json.dumps(
        {"type": type, "query": query, "topk": topk, "summary": summary},
        ensure_ascii=False,
        indent=2,
    )
    logger.debug(result)

    return result


@function_tool
async def tool_search_text_text(
    ctx: RunContextWrapper[_RagAgentContext],
    args: _TextSearchArgs,
) -> str:
    """Search text documents by a text query.

    Args:
        ctx (RunContextWrapper[_RagAgentContext]): Execution context.
        args (_TextSearchArgs): Search parameters.

    Raises:
        ValueError: Raised when the query string is missing.

    Returns:
        str: JSON string that summarizes the search results.
    """
    query = args.get("query")
    if not query:
        raise ValueError("query is required")

    response = ctx.context.client.query_text_text(query, _TOPK)
    return _format_response("text_text", query, _TOPK, response)


@function_tool
async def tool_search_text_image(
    ctx: RunContextWrapper[_RagAgentContext],
    args: _TextSearchArgs,
) -> str:
    """Search image documents by a text query.

    Args:
        ctx (RunContextWrapper[_RagAgentContext]): Execution context.
        args (_TextSearchArgs): Search parameters.

    Raises:
        ValueError: Raised when the query string is missing.

    Returns:
        str: JSON string that summarizes the search results.
    """
    query = args.get("query")
    if not query:
        raise ValueError("query is required")

    response = ctx.context.client.query_text_image(query, _TOPK)
    return _format_response("text_image", query, _TOPK, response)


@function_tool
async def tool_search_image_image(ctx: RunContextWrapper[_RagAgentContext]) -> str:
    """Search image documents based on an uploaded reference image.

    Args:
        ctx (RunContextWrapper[_RagAgentContext]): Execution context.

    Raises:
        ValueError: Raised when no reference image is registered.

    Returns:
        str: JSON string that summarizes the search results.
    """
    upload_id = ctx.context.upload_id
    if not upload_id:
        raise ValueError("upload_id is not provided in context")

    response = ctx.context.client.query_image_image(upload_id=upload_id, topk=_TOPK)
    return _format_response("image_image", upload_id, _TOPK, response)


@function_tool
async def tool_search_text_audio(
    ctx: RunContextWrapper[_RagAgentContext],
    args: _TextSearchArgs,
) -> str:
    """Search audio documents by a text query.

    Args:
        ctx (RunContextWrapper[_RagAgentContext]): Execution context.
        args (_TextSearchArgs): Search parameters.

    Raises:
        ValueError: Raised when the query string is missing.

    Returns:
        str: JSON string that summarizes the search results.
    """
    query = args.get("query")
    if not query:
        raise ValueError("query is required")

    response = ctx.context.client.query_text_audio(query, _TOPK)
    return _format_response("text_audio", query, _TOPK, response)


@function_tool
async def tool_search_audio_audio(ctx: RunContextWrapper[_RagAgentContext]) -> str:
    """Search audio documents based on an uploaded reference audio file.

    Args:
        ctx (RunContextWrapper[_RagAgentContext]): Execution context.

    Raises:
        ValueError: Raised when no reference audio is registered.

    Returns:
        str: JSON string that summarizes the search results.
    """
    upload_id = ctx.context.upload_id
    if not upload_id:
        raise ValueError("upload_id is not provided in context")

    response = ctx.context.client.query_audio_audio(upload_id=upload_id, topk=_TOPK)
    return _format_response("audio_audio", upload_id, _TOPK, response)


@function_tool
async def tool_search_text_video(
    ctx: RunContextWrapper[_RagAgentContext],
    args: _TextSearchArgs,
) -> str:
    """Search video documents by a text query.

    Args:
        ctx (RunContextWrapper[_RagAgentContext]): Execution context.
        args (_TextSearchArgs): Search parameters.

    Raises:
        ValueError: Raised when the query string is missing.

    Returns:
        str: JSON string that summarizes the search results.
    """
    query = args.get("query")
    if not query:
        raise ValueError("query is required")

    response = ctx.context.client.query_text_video(query, _TOPK)
    return _format_response("text_video", query, _TOPK, response)


@function_tool
async def tool_search_image_video(ctx: RunContextWrapper[_RagAgentContext]) -> str:
    """Search video documents using an uploaded reference image.

    Args:
        ctx (RunContextWrapper[_RagAgentContext]): Execution context.

    Raises:
        ValueError: Raised when no reference image is registered.

    Returns:
        str: JSON string that summarizes the search results.
    """
    upload_id = ctx.context.upload_id
    if not upload_id:
        raise ValueError("upload_id is not provided in context")

    response = ctx.context.client.query_image_video(upload_id=upload_id, topk=_TOPK)
    return _format_response("image_video", upload_id, _TOPK, response)


@function_tool
async def tool_search_audio_video(ctx: RunContextWrapper[_RagAgentContext]) -> str:
    """Search video documents using an uploaded reference audio file.

    Args:
        ctx (RunContextWrapper[_RagAgentContext]): Execution context.

    Raises:
        ValueError: Raised when no reference audio is registered.

    Returns:
        str: JSON string that summarizes the search results.
    """
    upload_id = ctx.context.upload_id
    if not upload_id:
        raise ValueError("upload_id is not provided in context")

    response = ctx.context.client.query_audio_video(upload_id=upload_id, topk=_TOPK)
    return _format_response("audio_video", upload_id, _TOPK, response)


@function_tool
async def tool_search_video_video(ctx: RunContextWrapper[_RagAgentContext]) -> str:
    """Search video documents using an uploaded reference video file.

    Args:
        ctx (RunContextWrapper[_RagAgentContext]): Execution context.

    Raises:
        ValueError: Raised when no reference video is registered.

    Returns:
        str: JSON string that summarizes the search results.
    """
    upload_id = ctx.context.upload_id
    if not upload_id:
        raise ValueError("upload_id is not provided in context")

    response = ctx.context.client.query_video_video(upload_id=upload_id, topk=_TOPK)
    return _format_response("video_video", upload_id, _TOPK, response)


_TOOLSET = [
    tool_search_text_text,
    tool_search_text_image,
    tool_search_image_image,
    tool_search_text_audio,
    tool_search_audio_audio,
    tool_search_text_video,
    tool_search_image_video,
    tool_search_audio_video,
    tool_search_video_video,
]


@dataclass(kw_only=True)
class RagAgentManager:
    """Manager that runs RAG searches using openai-agents."""

    client: RestAPIClient
    model: str

    def run(
        self,
        *,
        question: str,
        upload_id: Optional[str] = None,
        max_turns: int = 5,
    ) -> str:
        """Run the agent and return the final answer.

        Args:
            question (str): User question text.
            upload_id (Optional[str]): Upload id for the reference file.
            max_turns (int): Maximum number of agent turns.

        Raises:
            ValueError: Raised when the question is empty.
            AgentExecutionError: Raised when the agent execution fails.

        Returns:
            str: Final answer generated by the agent.
        """
        if question.strip() == "":
            raise ValueError("question must not be empty")

        logger.debug([tool.name for tool in _TOOLSET])
        agent = Agent(
            name="rag_assistant",
            instructions=(
                "You are a search agent. "
                "Always respond to the user in user language. "
                "Before answering, you MUST use the provided tools to search the knowledge base. "
                "If reference files are available, their upload ids are stored in upload_id. "
                "Use tools with the upload_id to search based on the reference files. "
                "When relevant documents are found, include the file paths in the answer. "
                "Do not include scores in the answer. "
                'If no relevant documents are found or an error occurs, reply with "No relevant documents were found." only.'
            ),
            tools=_TOOLSET,  # type: ignore
            model=self.model,
        )

        logger.info(f"upload id = {upload_id}")
        context = _RagAgentContext(
            client=self.client,
            upload_id=upload_id,
        )

        async def _run() -> str:
            result = await Runner.run(
                agent,
                input=question,
                max_turns=max_turns,
                context=context,
            )

            final = getattr(result, "final_output", None)
            if isinstance(final, str):
                return final
            if final is not None:
                return str(final)
            return ""

        try:
            return asyncio.run(_run())
        except Exception as e:
            logger.exception(e)
            raise AgentExecutionError(str(e)) from e

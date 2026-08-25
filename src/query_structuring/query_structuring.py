import datetime
from typing import Any, Optional, Type
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


def patch_youtube_loader():
    """Patch LangChain's YoutubeLoader with yt-dlp for reliable metadata extraction."""
    try:
        from langchain_community.document_loaders import YoutubeLoader
        from yt_dlp import YoutubeDL

        def patched_get_video_info(self):
            url = f"https://www.youtube.com/watch?v={self.video_id}"
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
            }
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
            return {
                "title": info.get("title"),
                "description": info.get("description"),
                "view_count": info.get("view_count"),
                "publish_date": info.get("upload_date"),
                "length": info.get("duration"),
                "author": info.get("uploader"),
                "thumbnail_url": info.get("thumbnail"),
            }

        YoutubeLoader._get_video_info = patched_get_video_info
        return True
    except ImportError:
        return False


class TutorialSearch(BaseModel):
    """Structured query model for searching tutorial videos database."""

    content_search: str = Field(
        ...,
        description="Similarity search query applied to video transcripts.",
    )
    title_search: str = Field(
        ...,
        description="Short search query containing keywords that may appear in the video title.",
    )
    min_view_count: Optional[int] = Field(
        None,
        description="Minimum view count filter. Only use if explicitly specified.",
    )
    max_view_count: Optional[int] = Field(
        None,
        description="Maximum view count filter. Only use if explicitly specified.",
    )
    earliest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Earliest publish date filter. Only use if explicitly specified.",
    )
    latest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Latest publish date filter. Only use if explicitly specified.",
    )
    min_length_sec: Optional[int] = Field(
        None,
        description="Minimum video length in seconds. Only use if explicitly specified.",
    )
    max_length_sec: Optional[int] = Field(
        None,
        description="Maximum video length in seconds. Only use if explicitly specified.",
    )

    def pretty_print(self) -> None:
        """Print non-None fields in a readable format."""
        for field_name, value in self.model_dump().items():
            if value is not None:
                print(f"{field_name}: {value}")


QUERY_STRUCTURING_SYSTEM_PROMPT = """You are an expert at converting user questions into structured database queries.

You have access to a database of tutorial videos about a software library
for building LLM-powered applications.

Given a user question, return a structured query optimized to retrieve
the most relevant results.

If there are acronyms or words you are not familiar with,
do not try to rephrase them.
"""

QUERY_STRUCTURING_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QUERY_STRUCTURING_SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def create_query_analyzer(
    llm: Optional[Any] = None,
    schema: Type[BaseModel] = TutorialSearch,
    prompt: Optional[ChatPromptTemplate] = None,
):
    """Create a structured query analyzer chain that parses natural questions into structured filter models."""
    if llm is None:
        llm = ChatOllama(model="llama3:latest", temperature=0)
    if prompt is None:
        prompt = QUERY_STRUCTURING_PROMPT

    structured_llm = llm.with_structured_output(schema)
    return prompt | structured_llm



from __future__ import annotations

from typing import Any, Dict, Generator, List, Optional, Sequence, Union

from enum import Enum

from urllib.parse import parse_qs, urlparse

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_NETLOCS = {
    "youtu.be",
    "m.youtube.com",
    "youtube.com",
    "www.youtube.com",
    "www.youtube-nocookie.com",
    "vid.plus",
}

class TranscriptFormat(Enum):
    """Output formats of transcripts from `YoutubeLoader`."""

    TEXT = "text"
    LINES = "lines"
    CHUNKS = "chunks"

class YoutubeLoaderFix(BaseLoader):
    """Load `YouTube` video transcripts."""

    def __init__(
        self,
        video_id: str,
        add_video_info: bool = False,
        language: Union[str, Sequence[str]] = "en",
        translation: Optional[str] = None,
        transcript_format: TranscriptFormat = TranscriptFormat.TEXT,
        continue_on_failure: bool = False,
        chunk_size_seconds: int = 120,
    ):
        """Initialize with YouTube video ID."""
        self.video_id = video_id
        self._metadata = {"source": video_id}
        self.add_video_info = add_video_info
        self.language = language
        if isinstance(language, str):
            self.language = [language]
        else:
            self.language = language
        self.translation = translation
        self.transcript_format = transcript_format
        self.continue_on_failure = continue_on_failure
        self.chunk_size_seconds = chunk_size_seconds

    @staticmethod
    def extract_video_id(youtube_url: str) -> str:
        """Extract video ID from common YouTube URLs."""
        video_id = _parse_video_id(youtube_url)
        if not video_id:
            raise ValueError(
                f'Could not determine the video ID for the URL "{youtube_url}".'
            )
        return video_id

    @classmethod
    def from_youtube_url(cls, youtube_url: str, **kwargs: Any) -> YoutubeLoaderFix:
        """Given a YouTube URL, construct a loader.
        See `YoutubeLoader()` constructor for a list of keyword arguments.
        """
        video_id = cls.extract_video_id(youtube_url)
        return cls(video_id, **kwargs)

    def _make_chunk_document(
        self, chunk_pieces: List[Dict], chunk_start_seconds: int
    ) -> Document:
        """Create Document from chunk of transcript pieces."""
        m, s = divmod(chunk_start_seconds, 60)
        h, m = divmod(m, 60)
        return Document(
            page_content=" ".join(
                map(lambda chunk_piece: chunk_piece["text"].strip(" "), chunk_pieces)
            ),
            metadata={
                **self._metadata,
                "start_seconds": chunk_start_seconds,
                "start_timestamp": f"{h:02d}:{m:02d}:{s:02d}",
                "source":
                # replace video ID with URL to start time
                f"https://www.youtube.com/watch?v={self.video_id}"
                f"&t={chunk_start_seconds}s",
            },
        )

    def _get_transcript_chunks(
        self, transcript_pieces: List[Dict]
    ) -> Generator[Document, None, None]:
        chunk_pieces: List[Dict[str, Any]] = []
        chunk_start_seconds = 0
        chunk_time_limit = self.chunk_size_seconds
        for transcript_piece in transcript_pieces:
            piece_end = transcript_piece["start"] + transcript_piece["duration"]
            if piece_end > chunk_time_limit:
                if chunk_pieces:
                    yield self._make_chunk_document(chunk_pieces, chunk_start_seconds)
                chunk_pieces = []
                chunk_start_seconds = chunk_time_limit
                chunk_time_limit += self.chunk_size_seconds

            chunk_pieces.append(transcript_piece)

        if len(chunk_pieces) > 0:
            yield self._make_chunk_document(chunk_pieces, chunk_start_seconds)

    def load(self) -> List[Document]:
        """Load YouTube transcripts into `Document` objects."""
        try:
            from youtube_transcript_api import (
                NoTranscriptFound,
                TranscriptsDisabled,
                YouTubeTranscriptApi,
            )
        except ImportError:
            raise ImportError(
                'Could not import "youtube_transcript_api" Python package. '
                "Please install it with `pip install youtube-transcript-api`."
            )

        if self.add_video_info:
            # Get more video meta info
            # Such as title, description, thumbnail url, publish_date
            video_info = self._get_video_info()
            self._metadata.update(video_info)

        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(self.video_id)
        except TranscriptsDisabled:
            return []

        try:
            transcript = transcript_list.find_transcript(self.language)
        except NoTranscriptFound:
            transcript = transcript_list.find_transcript(["en"])

        if self.translation is not None:
            transcript = transcript.translate(self.translation)

        # transcript_pieces: List[Dict[str, Any]] = transcript.fetch()
        
        print(f"[DEBUG] Attempting to fetch transcript for video: {self.video_id}")
        try:
            
            raw_transcript = transcript.fetch()
            transcript_pieces = [
                {
                    "text": piece.text,
                    "start": piece.start,
                    "duration": piece.duration,
                }
                for piece in raw_transcript
            ]
            
            print(f"[DEBUG] Fetched {len(transcript_pieces)} transcript entries.")
            print(f"[DEBUG] First few entries: {transcript_pieces[:3]}")
        except Exception as e:
            print(f"[ERROR] transcript.fetch() failed for video {self.video_id}: {e}")
            return []

        if self.transcript_format == TranscriptFormat.TEXT:
            # * FIX: New pytubefix version has transcript_piece.text instead of transcript_piece["text"]
            try: 
                transcript = " ".join(
                    map(
                        lambda transcript_piece: transcript_piece.text.strip(" "),
                        transcript_pieces,
                    )
                )
            except:
                transcript = " ".join(
                    map(
                        lambda transcript_piece: transcript_piece["text"].strip(" "),
                        transcript_pieces,
                    )
                )
            return [Document(page_content=transcript, metadata=self._metadata)]
        elif self.transcript_format == TranscriptFormat.LINES:
            
            try:
                transcript = list(
                    map(
                        lambda transcript_piece: Document(
                            page_content=transcript_piece.text.strip(" "),
                            metadata=dict(
                                filter(
                                    lambda item: item[0] != "text", transcript_piece.items()
                                )
                            ),
                        ),
                        transcript_pieces,
                    )
                )
            except:
                transcript = list(
                    map(
                        lambda transcript_piece: Document(
                            page_content=transcript_piece["text"].strip(" "),
                            metadata=dict(
                                filter(
                                    lambda item: item[0] != "text", transcript_piece.items()
                                )
                            ),
                        ),
                        transcript_pieces,
                    )
                )            
            return transcript
        elif self.transcript_format == TranscriptFormat.CHUNKS:
            return list(self._get_transcript_chunks(transcript_pieces))

        else:
            raise ValueError("Unknown transcript format.")

    def _get_video_info(self) -> Dict:
        """Get important video information.

        Components include:
            - title
            - description
            - thumbnail URL,
            - publish_date
            - channel author
            - and more.
        """
        try:
            from pytubefix import YouTube

        except ImportError:
            raise ImportError(
                'Could not import "pytubefix" Python package. '
                "Please install it with `pip install pytubefix`."
            )
        yt = YouTube(
            url=f"https://www.youtube.com/watch?v={self.video_id}", 
            client='WEB',
        )
        video_info = {
            "title": yt.title or "Unknown",
            "description": yt.description or "Unknown",
            "view_count": yt.views or 0,
            "thumbnail_url": yt.thumbnail_url or "Unknown",
            "publish_date": yt.publish_date.strftime("%Y-%m-%d %H:%M:%S")
            if yt.publish_date
            else "Unknown",
            "length": yt.length or 0,
            "author": yt.author or "Unknown",
        }
        return video_info
    
    

def _parse_video_id(url: str) -> Optional[str]:
    """Parse a YouTube URL and return the video ID if valid, otherwise None."""
    parsed_url = urlparse(url)

    if parsed_url.scheme not in ALLOWED_SCHEMES:
        return None

    if parsed_url.netloc not in ALLOWED_NETLOCS:
        return None

    path = parsed_url.path

    if path.endswith("/watch"):
        query = parsed_url.query
        parsed_query = parse_qs(query)
        if "v" in parsed_query:
            ids = parsed_query["v"]
            video_id = ids if isinstance(ids, str) else ids[0]
        else:
            return None
    else:
        path = parsed_url.path.lstrip("/")
        video_id = path.split("/")[-1]

    if len(video_id) != 11:  # Video IDs are 11 characters long
        return None

    return video_id


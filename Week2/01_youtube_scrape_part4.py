from typing import List
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_core.documents import Document


class YoutubeLoaderFix:
    """
    Custom YouTube loader that fetches video transcripts using youtube-transcript-api
    and returns them as LangChain Document objects.
    """

    def __init__(self, video_url: str):
        self.video_url = video_url
        self.video_id = self._extract_video_id(video_url)

    def _extract_video_id(self, url: str) -> str:
        """
        Extract the YouTube video ID from a URL.
        Supports both long-form and short-form links.
        """
        if "v=" in url:
            return url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0]
        else:
            raise ValueError(f"Could not extract video ID from URL: {url}")

    def load(self) -> List[Document]:
        """
        Loads the transcript for the video and returns a list of LangChain-compatible Documents.
        If no transcript is available, returns an empty list.
        """
        try:
            transcript = YouTubeTranscriptApi.get_transcript(self.video_id)
            full_text = " ".join([entry["text"] for entry in transcript])

            return [Document(
                page_content=full_text,
                metadata={
                    "source": "youtube",
                    "video_id": self.video_id,
                    "video_url": self.video_url
                }
            )]

        except (TranscriptsDisabled, NoTranscriptFound) as e:
            print(f"[ERROR] Transcript unavailable for video {self.video_id}: {e}")
            return []
        except Exception as e:
            print(f"[ERROR] Failed to fetch transcript for {self.video_id}: {e}")
            return []
    

# url = "https://www.youtube.com/watch?v=hMtOmZYLEOQ"
# loader = YoutubeLoaderFix(url)
YoutubeLoaderFix("https://www.youtube.com/watch?v=ATlMK7ln5Dc")  
docs = loader.load()

if docs:
    print(docs[0].page_content[:500])  # Print preview
    print(docs[0].metadata)
else:
    print("No transcript found.")
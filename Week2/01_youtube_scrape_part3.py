import pandas as pd
import yaml
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from googleapiclient.discovery import build

# --- Load API Key ---
def load_api_key(path="../credentials.yml"):
    with open(path, "r") as f:
        keys = yaml.safe_load(f)
    return keys["youtube"]

# --- Search YouTube Videos ---
def search_youtube_videos(topic, api_key, max_results=5):
    youtube = build("youtube", "v3", developerKey=api_key)
    request = youtube.search().list(
        q=topic,
        part="id,snippet",
        maxResults=max_results,
        type="video"
    )
    response = request.execute()
    return [item["id"]["videoId"] for item in response["items"]]

# --- Fetch Transcript ---
def fetch_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except (TranscriptsDisabled, NoTranscriptFound) as e:
        print(f"[SKIPPED] No transcript for {video_id}: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error for {video_id}: {e}")
        return None

# --- Main ---
def main():
    topic = "Python tutorial for beginners"
    api_key = load_api_key()
    video_ids = search_youtube_videos(topic, api_key)
    video_ids = ['yXWw0_UfSFg']
    rows = []
    for vid in video_ids:
        print(f"[FETCHING] {vid}")
        transcript = fetch_transcript(vid)
        if transcript:
            full_text = " ".join([t["text"] for t in transcript])
            rows.append({
                "video_id": vid,
                "video_url": f"https://www.youtube.com/watch?v={vid}",
                "transcript": full_text
            })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv("youtube_transcripts.csv", index=False)
        print("[✅] Transcripts saved to youtube_transcripts.csv")
    else:
        print("[INFO] No transcripts available from these videos.")

if __name__ == "__main__":
    main()

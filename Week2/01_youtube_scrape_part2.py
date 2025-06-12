# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# RETRIEVAL-AUGMENTED GENERATION (RAG)

# GOAL: Collect Expertise on a Topic that's relevant to your business application

# 1.0 IMPORTS 

# * FIX: pytube is currently broken. This patch requires pytubefix
# !pip install pytubefix

from langchain_pytubefix import YoutubeLoaderFix
from googleapiclient.discovery import build
import yaml
import os
import pandas as pd
from tqdm import tqdm
from pprint import pprint
import streamlit as st 

# 2.0 YOUTUBE API KEY SETUP 
PATH_CREDENTIALS = '../credentials.yml'
os.environ['YOUTUBE_API_KEY'] = yaml.safe_load(open(PATH_CREDENTIALS))['youtube'] 

# 3.0 VIDEO TRANSCRIPT SCRAPING FUNCTIONS
def search_videos(topic, api_key, max_results=20):
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.search().list(
        q=topic,
        part='id,snippet',
        maxResults=max_results,
        type='video'
    )
    response = request.execute()
    video_ids = [item['id']['videoId'] for item in response['items']]
    return video_ids

def load_video(video_id, add_video_info=False):
    url = f'https://www.youtube.com/watch?v={video_id}'
    try:
        loader = YoutubeLoaderFix.from_youtube_url(
            url, 
            add_video_info=add_video_info,
        )
        print(f"[DEBUG] Attempting to fetch transcript for video: {video_id}")
        docs = loader.load()
        print(f"[DEBUG] DOCS returned: {docs}")

        if not docs:
            print(f"[WARNING] No transcript returned for video: {video_id}")
            return None

        doc = docs[0]
        doc_df = pd.DataFrame([doc.metadata])
        doc_df['video_url'] = url
        doc_df['page_content'] = doc.page_content
        return doc_df

    except Exception as e:
        print(f"[ERROR] Failed to load transcript for video {video_id}: {e}")
        return None

# 4.0 SCRAPE YOUTUBE VIDEOS TRANSCRIPTS
TOPIC = "Social Media Brand Strategy Tips"
video_ids = search_videos(
    topic=TOPIC, 
    api_key=os.environ['YOUTUBE_API_KEY'], 
    max_results=5
)

# Try loading one video safely
video = load_video(video_ids[3], add_video_info=False)
if video is not None:
    pprint(video['page_content'][0])
else:
    print(f"No transcript available for video: {video_ids[3]}")

# 5.0 PROCESS ALL VIDEOS
videos = []
for video_id in tqdm(video_ids, desc="Processing videos"):
    try:
        video = load_video(video_id, add_video_info=False)
        if video is not None:
            pprint(video['page_content'][0])
            videos.append(video)
        else:
            print(f"[SKIPPED] No transcript for {video_id}")
    except Exception as e:
        print(f"Skipping video {video_id} due to error: {e}")

# 6.0 SAVE VIDEOS TO CSV
if videos:
    videos_df = pd.concat(videos, ignore_index=True)
    videos_df.to_csv('data/youtube_videos.csv', index=False)
    print("[SUCCESS] Transcripts saved to data/youtube_videos.csv")
    pprint(videos_df['page_content'][0])
    print("First video URL:", videos_df['video_url'][0])
else:
    print("[INFO] No transcripts found — nothing to save.")
import os
import tempfile
import whisper
from pytubefix import YouTube
from pytubefix.cli import on_progress
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_trascription_from_youtube(url):
    if not os.path.exists("transcription.txt"):
        youtube = YouTube(url, on_progress_callback = on_progress)
        audio = youtube.streams.filter(only_audio=True).first()

        whisper_model = whisper.load_model("base")

        with tempfile.TemporaryDirectory() as tmpdir:
            print("downloading the video...")
            file = audio.download(output_path=tmpdir)
            print("the video was download successfully")
            transcription = whisper_model.transcribe(file, fp16=False)["text"].strip()

            with open("transcription.txt", "w") as file:
                file.write(transcription)

def extract_chunks_from_youtube(url):
    extract_trascription_from_youtube(url)
    loader = TextLoader("transcription.txt")
    text_documents = loader.load()

    # Define a splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(text_documents)
    return chunks


def get_youtube_title(url):
    youtube = YouTube(url)
    return youtube.title
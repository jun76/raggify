from raggify.ingest import ingest_path_list
from raggify.retrieve import query_text_audio

paths = [
    "/path/to/sound.mp3",
    "/path/to/sound.wav",
    "/path/to/sound.flac",
    "/path/to/sound.ogg",
]

ingest_path_list(paths)

nodes = query_text_audio(query="phone call")

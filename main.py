# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 00:55:37 2023

@author: caotc
"""
import os
import openai
import yt_dlp as youtube_dl
import openai
import glob
from yt_dlp import DownloadError
import docarray


#Key openAI
openai_api_key = os.getenv("OPENAI_API_KEY")

#Download the YouTube Video and convert mp3
youtube_url = "https://www.youtube.com/watch?v=znq3ql6wqnE&t=18s"
# Directory to store the downloaded video
output_dir = "files/audio/"

# Config for youtube-dl
ydl_config = {
    "format": "bestaudio/best",
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }
    ],
    "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
    "verbose": True
}




# Check if the output directory exists, if not create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Print a message indicating which video is being downloaded
print(f"Downloading video from {youtube_url}")
# Attempt to download the video using the specified configuration
# If a DownloadError occurs, attempt to download the video again
try:
    with youtube_dl.YoutubeDL(ydl_config) as ydl:
        ydl.download([youtube_url])
except DownloadError:
    with youtube_dl.YoutubeDL(ydl_config) as ydl:
        ydl.download([youtube_url])



# Task: Find the audio file in the output directory

# Find all the audio files in the output directory
audio_file = glob.glob(os.path.join(output_dir, "*.mp3"))
# Select the first audio file in the list
audio_filename = audio_file[0]
# Print the name of the selected audio file
print(audio_filename)


# Task: Transcribe the Video using Whisper
# Define function parameters
audio_file = audio_filename
output_file = "files/transcripts/transcript.txt"
model = "whisper-1"

# Transcribe the audio file to text using OpenAI API
print("converting audio to text...")

with open (audio_file, "rb") as audio:
    response = openai.Audio.transcribe(model, audio)
    
# Extract the transcript from the response
transcript = (response["text"])


# If an output file is specified, save the transcript to a .txt file

if output_file is not None:
    # Create the directory for the output file if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Write the transcript to the output file
    with open(output_file, "w") as file:
        file.write(transcript)

# Print the transcript to the console to verify it worked 
print(transcript)




#Task:  Create a TextLoader using LangChain
# Import the TextLoader class from the langchain.document_loaders module
from langchain.document_loaders import TextLoader

# Create a new instance of the TextLoader class, specifying the directory containing the text files
loader = TextLoader("./files/transcripts/transcript.txt")

# Load the documents from the specified directory using the TextLoader instance

docs = loader.load()


# Show the first element of docs to verify it has been loaded 
print(docs[0])

#Task: Creating an In-Memory Vector Store
#Import the tiktoken package
import tiktoken

#Task: Create the Document Search
# Import the RetrievalQA class from the langchain.chains module
from langchain.chains import RetrievalQA

# Import the ChatOpenAI class from the langchain.chat_models module
from langchain.chat_models import ChatOpenAI

# Import the DocArrayInMemorySearch class from the langchain.vectorstores module
from langchain.vectorstores import DocArrayInMemorySearch

# Import the OpenAIEmbeddings class from the langchain.embeddings module
from langchain.embeddings import OpenAIEmbeddings


# Create a new DocArrayInMemorySearch instance from the specified documents and embeddings
db = DocArrayInMemorySearch.from_documents(
    docs, 
    OpenAIEmbeddings())


# Convert the DocArrayInMemorySearch instance to a retriever
retriever = db.as_retriever()

# Create a new ChatOpenAI instance with a temperature of 0.0
llm = ChatOpenAI(temperature=0.0)


# Create a new RetrievalQA instance with the specified parameters
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True)

#Task: Create the Queries
# Set the query to be used for the QA system
query = "¿De qué se trata este tutorial?"

# Run the query through the RetrievalQA instance and store the response
response = qa_stuff.run(query)

# Print the response to the console
print(response)


# Set the query to be used for the QA system
query = "¿Cuál es la diferencia entre un conjunto de entrenamiento y un conjunto de prueba?"

# Run the query through the RetrievalQA instance and store the response
response = qa_stuff.run(query)


# Print the response to the console
print(response)


# Set the query to be used for the QA system
query = "¿Quién debería ver esta lección?"

# Run the query through the RetrievalQA instance and store the response
response = qa_stuff.run(query)


# Print the response to the console
print(response)


# Set the query to be used for the QA system
query = "¿Quién debería ver esta lección?"

# Run the query through the RetrievalQA instance and store the response
response = qa_stuff.run(query)


# Print the response to the console
print(response)# Set the query to be used for the QA system
query = "¿Quién es el mejor equipo de fútbol del mundo?"

# Run the query through the RetrievalQA instance and store the response
response = qa_stuff.run(query)


# Print the response to the console
print(response)


# Print the response to the console
print(response)# Set the query to be used for the QA system
query = "¿Cuánto mide la circunferencia de la tierra?"

# Run the query through the RetrievalQA instance and store the response
response = qa_stuff.run(query)


# Print the response to the console
print(response)


# Print the response to the console
print(response)# Set the query to be used for the QA system
query = "Dí una conclusión del video"

# Run the query through the RetrievalQA instance and store the response
response = qa_stuff.run(query)


# Print the response to the console
print(response)








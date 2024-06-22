# Alloy Voice Assistant

AI Assistant that can read your screen and respond to your queries

You need an `OPENAI_API_KEY` and a `GROQ_API_KEY` to run this code. Store them in a `.env` file in the root directory of the project, or set them as environment variables.

This project was written to run on a mac.

## Installation

```
brew install portaudio
```

Create a virtual environment, update pip, and install the required packages:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Run the assistant:

```
python3 assistant.py
```

## Footnotes

This project is a fork of <https://github.com/svpino/alloy-voice-assistant>

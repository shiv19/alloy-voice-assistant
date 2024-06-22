import base64
from io import BytesIO
import time
import logging
import os
import threading
from collections import deque

from groq import Groq
import openai
from PIL import ImageGrab
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError, RequestError, AudioData

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Assistant:
    def __init__(self, model):
        self.chat_history = ChatMessageHistory()
        self.chain = self._create_inference_chain(model)
        self.command_buffer = deque(maxlen=5)  # Store last 5 commands
        self.last_command_time = 0
        self.debounce_time = 2  # Seconds to wait between commands
        self.min_word_count = 3  # Minimum number of words for a valid command

    @staticmethod
    def play_sound(path):
        os.system(f'afplay "{path}"')

    def answer(self, prompt):
        if not self._is_valid_command(prompt):
            return

        current_time = time.time()
        if current_time - self.last_command_time < self.debounce_time:
            logging.info(f"Debounced prompt: {prompt}")
            return

        self.last_command_time = current_time
        logging.info(f"Processing prompt: {prompt}")

        # Play sound to indicate API call
        threading.Thread(target=self.play_sound, args=('/System/Library/Sounds/Blow.aiff',)).start()

        try:
            screenshot = ImageGrab.grab()
            screenshot_rgb = screenshot.convert('RGB')
            buffered = BytesIO()
            screenshot_rgb.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            start_time = time.time()
            response = self.chain.invoke({
                "prompt": prompt,
                "image_base64": img_str,
                "chat_history": self.chat_history.messages
            })
            end_time = time.time()

            self.chat_history.add_user_message(prompt)
            self.chat_history.add_ai_message(response)

            logging.info(f"Generated response: {response}")
            logging.info(f"Response time: {end_time - start_time:.2f} seconds")

            if response:
                self._tts(response)
        except Exception as e:
            logging.error(f"Error in answer method: {str(e)}")

    def _is_valid_command(self, prompt):
        if not prompt or len(prompt.split()) < self.min_word_count:
            return False

        # Check for repetition
        if prompt in self.command_buffer:
            logging.warning(f"Repeated prompt detected: {prompt}")
            return False

        self.command_buffer.append(prompt)
        return True

    def _tts(self, response):
        try:
            player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)

            with openai.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="alloy",
                response_format="pcm",
                input=response,
            ) as stream:
                for chunk in stream.iter_bytes(chunk_size=1024):
                    player.write(chunk)
        except Exception as e:
            logging.error(f"Error in text-to-speech: {str(e)}")

    def _create_inference_chain(self, model):
        SYSTEM_PROMPT = """
        You are a witty assistant that will use the chat history and the screenshot
        provided by the user to answer its questions.

        Use few words on your answers. Go straight to the point. Do not use any
        emoticons or emojis. Do not ask the user any questions.

        Be friendly and helpful. Show some personality. Do not be too formal.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )

        return (
            RunnablePassthrough.assign(chat_history=lambda x: x["chat_history"])
            | prompt_template
            | model
            | StrOutputParser()
        )

def filter_speech(text):
    filler_words = {'um', 'uh', 'like', 'you know', 'actually', 'basically', 'literally', 'so', 'well', 'okay'}
    words = text.lower().split()
    filtered_words = [word for word in words if word not in filler_words]
    return ' '.join(filtered_words)

def remove_repeated_phrases(text):
    words = text.split()
    result = []
    for i in range(len(words)):
        if i == 0 or words[i] != words[i-1]:
            result.append(words[i])
    return ' '.join(result)

def audio_callback(recognizer, audio, assistant):
    try:
        audio: AudioData = audio
        with open("audio-file.wav", "wb") as f:
            f.write(audio.get_wav_data())

        client = Groq()
        filename = os.path.dirname(__file__) + "/audio-file.wav"

        with open(filename, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(filename, file.read()),
                model="whisper-large-v3",
                response_format="json",
                language="en",
                temperature=0.0
            )

        response = transcription.text
        filtered_response = filter_speech(response)
        clean_response = remove_repeated_phrases(filtered_response)

        if clean_response:
            logging.info(f"Recognized speech: {clean_response}")
            assistant.answer(clean_response)
        else:
            logging.info(f"Ignored invalid or short phrase: {response}")

    except UnknownValueError:
        logging.warning("Speech recognition could not understand audio")
    except RequestError as e:
        logging.error(f"Could not request results from speech recognition service; {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error in audio callback: {str(e)}")

def main():
    model = ChatOpenAI(model="gpt-4o")
    assistant = Assistant(model)

    recognizer = Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    microphone = Microphone()

    print("Listening for voice commands. Press Ctrl+C to exit.")
    logging.info("Starting voice command listener")

    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)

        stop_listening = recognizer.listen_in_background(
            microphone,
            lambda recognizer, audio: audio_callback(recognizer, audio, assistant),
            phrase_time_limit=30
        )

        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping...")
    except Exception as e:
        logging.error(f"Unexpected error in main loop: {str(e)}")
    finally:
        logging.info("Shutting down")
        if 'stop_listening' in locals():
            stop_listening(wait_for_stop=False)

if __name__ == "__main__":
    main()
import base64
from io import BytesIO
import time
import logging
import os
import threading

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
from speech_recognition import Microphone, Recognizer, UnknownValueError, RequestError

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Assistant:
    def __init__(self, model):
        self.chat_history = ChatMessageHistory()
        self.chain = self._create_inference_chain(model)
        self.last_prompt = ""
        self.repeat_count = 0
        self.last_command_time = 0   # Track the time of the last command

    @staticmethod
    def acknowledge_prompt():
        os.system('say "One moment please. Taking a screenshot and sharing it with the assistant."')

    @staticmethod
    def play_sound(path):
        os.system(f'afplay "{path}"')

    def answer(self, prompt):
        if not prompt:
            return

        current_time = time.time()
        time_since_last_command = current_time - self.last_command_time

        # Check for repeated prompts
        if prompt == self.last_prompt:
            self.repeat_count += 1
            if self.repeat_count > 2:
                logging.warning(f"Repeated prompt detected: {prompt}")
                return
        else:
            self.last_prompt = prompt
            self.repeat_count = 0

        if time_since_last_command < 2:  # Wait at least 2 seconds between processing commands
            logging.info(f"Debounced prompt: {prompt}")
            return
        self.last_command_time = current_time

        logging.info(f"Received prompt: {prompt}")

        # Play a blow sound to indicate that the assistant API call is about to be made
        play_blow_sound = threading.Thread(target=Assistant.play_sound, args=('/System/Library/Sounds/Blow.aiff',))
        play_blow_sound.start()

        start_time = time.time()

        try:
            # ack_thread = threading.Thread(target=Assistant.acknowledge_prompt)
            # ack_thread.start()
            # Take a screenshot and convert to RGB
            screenshot = ImageGrab.grab()
            screenshot_rgb = screenshot.convert('RGB')
            buffered = BytesIO()
            screenshot_rgb.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            response = self.chain.invoke({
                "prompt": prompt,
                "image_base64": img_str,
                "chat_history": self.chat_history.messages
            })

            # Add the exchange to chat history
            self.chat_history.add_user_message(prompt)
            self.chat_history.add_ai_message(response)

            logging.info(f"Generated response: {response}")
            end_time = time.time()  # End the timer
            duration = end_time - start_time  # Calculate the duration
            logging.info(f"Response time: {duration:.2f} seconds")

            if response:
                self._tts(response)
        except Exception as e:
            logging.error(f"Error in answer method: {str(e)}")

    def _tts(self, response):
        try:
            # Use the macOS 'say' command to generate text-to-speech
            os.system(f'say "{response}"')

            # Use the OpenAI API to generate text-to-speech
            # player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)

            # with openai.audio.speech.with_streaming_response.create(
            #     model="tts-1",
            #     voice="alloy",
            #     response_format="pcm",
            #     input=response,
            # ) as stream:
            #     for chunk in stream.iter_bytes(chunk_size=1024):
            #         player.write(chunk)
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

model = ChatOpenAI(model="gpt-4o")
assistant = Assistant(model)

def audio_callback(recognizer, audio):
    try:
        # Recognize speech using the Whisper API
        response = recognizer.recognize_whisper(audio, model="base", language="english")

        # Simple noise and filler word filtering, and validation
        prompt = filter_speech(response)

        # Check if the prompt is non-trivial and remove repeated phrases
        if is_valid_prompt(prompt):
            clean_prompt = remove_repeated_phrases(prompt)
            logging.info(f"Recognized speech: {clean_prompt}")
            assistant.answer(clean_prompt)
        else:
            logging.info(f"Ignored invalid or short phrase: {response}")

    except UnknownValueError:
        logging.warning("Speech recognition could not understand audio")
    except RequestError as e:
        logging.error(f"Could not request results from speech recognition service; {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error in audio callback: {str(e)}")

def filter_speech(response):
    # A function to clean up and filter recognized speech text
    filler_words = {'um', 'uh', 'like', 'you know', 'actually', 'basically', 'literally', 'so', 'well', 'okay'}
    words = [word for word in response.lower().split() if word not in filler_words]
    return ' '.join(words)

def is_valid_prompt(prompt):
    # A function to check if a prompt is valid
    min_word_count = 3
    return len(prompt.split()) >= min_word_count and not contains_repetition(prompt)

def contains_repetition(text):
    # Check for repeated phrases
    words = text.split()
    for i in range(len(words) - 1):
        phrase = ' '.join(words[:i+1])
        remainder = ' '.join(words[i+1:])
        if remainder.startswith(phrase):
            return True
    return False

def remove_repeated_phrases(text):
    # Remove repeated phrases from text
    words = text.split()
    buffer = []
    seen_phrases = set()
    for i in range(len(words)):
        phrase = ' '.join(words[:i+1])
        if phrase not in seen_phrases:
            seen_phrases.add(phrase)
            buffer.append(words[i])
    return ' '.join(buffer)

recognizer = Recognizer()
microphone = Microphone()

print("Listening for voice commands. Press Ctrl+C to exit.")
logging.info("Starting voice command listener")

try:
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)

    stop_listening = recognizer.listen_in_background(microphone, audio_callback, phrase_time_limit=15)

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
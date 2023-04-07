GPT Talkbot
----
GPT Talkbot is a locally executable chatbot execution environment using GPT-based models, with the ability to listen to speech with Whisper and speak with VOICEVOX. It can be executed in a local environment using a model fine-tuned with a proprietary dataset. Facial expressions according to the execution state can be sent via WebSocket.

## Setup
1. Install required packages using `requirements.txt`:
```sh
pip install -r requirements.txt
```

2. Install code formatter (Black) and linters (Flake8, isort):
```sh
pip install black flake8 isort
```

3. Install Text To Speech Engine:

Currentry only supporting [VOICEVOX](https://voicevox.hiroshiba.jp/)

## Run

```
# Start VOICEVOX Engine
cd path/to/gpt-talkbot
python -m talkbot
```

### Run Console
`python -m talkbot console`

### Components
Each component in the TalkBot system can operate independently, allowing for modular development and integration. They communicate with each other using the message-stream component as an event bus.

### message-stream

`python -m talkbot message-stream`

Serves as a simple and transparent event bus, enabling pub-sub communication between other components. It is essential for the operation of all components, except for model training.

### Audio To Message

`python -m talkbot audio-to-message`

This component converts audio input from the microphone into text messages using the OpenAI Whisper ASR model. It processes the audio data in real-time, transcribes it, and sends the transcribed text as a message to the message-stream component. The transcription process involves filtering out unwanted segments based on a set of configurable parameters.

### Chat Engine

`python -m talkbot chat-engine`

This component is an async chatbot using T5 from Hugging Face's Transformers. It supports special commands (date, weather, news) and maintains conversation history for context-aware responses.

- T5 model for responses
- Special commands: date, weather, news
- Conversation history

### Message To Speak

`python -m talkbot message-to-speak`

This component converts text messages into speech using the VOICEVOX engine. It processes text messages received from the message-stream component, converts them into speech, and plays the audio through the system's speakers. It also supports converting English text to Kana using an English-to-Kana dictionary, if provided.

### Neos Connector

`python -m talkbot neos-connector`

This component connects to Neos and sends messages to the chat engine. It listens for state messages from the message-stream and broadcasts expressions to Neos based on the state of the components. It also maintains a WebSocket server to communicate with Neos.

### Prune

`python -m talkbot prune`

This component is used to prune a model to reduce its size. It applies L1 unstructured pruning and saves the pruned model and tokenizer to the specified directory. The pruning amount can be configured.

### Train

This component trains a model for the chatbot using the GPT-2 architecture from Hugging Face's Transformers. It generates training data by reading a CSV file and trains the model using a data collator, training arguments, and callbacks. After each epoch, it tests the model and saves the state, model, and tokenizer to the specified output directory.

- GPT-2 model
- T5 tokenizer
- Training from CSV file
- Callbacks for testing and saving
- ZeroMQ sockets communication

### Bridge

`python -m talkbot bridge`

This component bridges two talkbot instances, enabling communication between them. It listens for messages from one instance and forwards them to the other instance after removing specified patterns. The process is reversed to enable bidirectional communication.

- ZeroMQ sockets communication
- Message forwarding
- Pattern removal
- Configurable sleep duration

## Customize
Create a `config/local.yml` file to store your local settings. This file will be loaded automatically and merged with the default settings. You can also use the `--config` option followed by a path to a custom configuration file.

## Training
1. Create Dataset

Create a CSV file with the following format:

```csv
User Input, AI Response
```

Ensure that the file is UTF-8 encoded and uses the appropriate line endings for LF. A minimum of 500 data samples is recommended for training.

2. Setup config/local.yml
3. Run the training script: python -m talkbot train

## Contributing

We follow the PEP 8 coding style guidelines and use Black, Flake8, and isort to ensure consistent code formatting. Please ensure that your contributions adhere to these guidelines. Additional information on how to contribute, such as running tests and updating documentation, will be provided in the future.


## License
See [LICENSE](LICENSE) for details.

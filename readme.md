# AI Agent for QA on PDF 

A Retrieval-Augmented Generation (RAG) agent that processes PDF files, answers questions, and posts results to Slack.

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Project Structure](#project-structure)

## Features

- PDF text extraction and processing
- Vector store for efficient text retrieval using FAISS
- OpenAI integration for question answering
- Slack integration for posting results
- Configurable settings for easy customization

## Prerequisites

- OpenAI API key
- Slack Bot Token
- Slack channel name/ID


## Installation

1. Clone the repository:
   ```
   git clone https://github.com/bvenkatesh-ai/slackbot.git
   cd slackbot
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
## Configuration
1. Set up environment variables:
   Create a `.env` file in the root directory and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   SLACK_BOT_TOKEN=your_slack_bot_token
   ```
2. Configure the config file with desired paramaters

   You can modify the `config.yaml` file to adjust various settings such as OpenAI model, chunk size, and logging level. You must add channel name/ID

## Usage
1. Using cmd line 
Run the main script with the following command:

```
python main.py path/to/your/pdf path/to/questions.txt 
```

- `path/to/your/pdf`: Path to the PDF file you want to process
- `questions.txt`: A text file containing questions, one per line
2. Using streamlit UI
```
streamlit run demo.py
```
You will get a UI, where you will upload pdf file and add questions separated by ,
## Project Structure

```
slackbot/
├── .env
├── config.yaml
├── requirements.txt
├── README.md
├── main.py
├── demo.py
├── ai_agent/
│   ├── __init__.py
|   ├── ai_agent.py
│   └── src/
│       ├── pdf_processor.py
│       ├── vector_store.py
│       ├── openai_integration.py
│       ├── slack_integration.py
│       └── __init__.py
```


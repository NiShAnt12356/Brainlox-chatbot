# Brainlox Technical Courses Chatbot

## Overview
This project is an AI-powered chatbot for Brainlox technical courses. It utilizes LangChain, Hugging Face embeddings, FAISS vector storage, and Google Generative AI to retrieve relevant course-related information based on user queries. The chatbot is implemented as a Flask RESTful API.

## Features
- Web scraping of Brainlox course pages
- Text chunking and vector storage using FAISS
- Query retrieval using similarity search
- Contextual question answering with chat history
- Flask API for real-time interaction

## Installation
### Prerequisites
Ensure you have Python 3.8+ installed along with the required libraries. You can install dependencies using:
```bash
pip install -r requirements.txt
```

### Running the Chatbot
1. Set up environment variables using a `.env` file.
2. Run the script:
```bash
python project.py
```

## API Usage
The chatbot API provides an endpoint:
- **POST /ask**: Send a JSON payload with a `query` to get a response.

Example request:
```json
{
  "query": "What is the Java Coding Summer Camp?"
}
```


Example response:
```json
{
  "response": "The Java Coding Summer Camp is a course designed for beginners to learn Java programming."
}
```


## Contributing
Feel free to improve the chatbot by enhancing retrieval mechanisms or integrating additional LLM models. Fork the repository, make changes, and submit a pull request.

## Contact
For any issues or suggestions, reach out to the project contributors.


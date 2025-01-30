# Multiple-PDF-Chatbot

![image](https://github.com/user-attachments/assets/f64cacb6-cc8b-4fad-88f2-cfa6aa136118)

This Streamlit application allows users to upload multiple PDF files, process them, and interact with the content using a chatbot powered by Google's Generative AI. Users can ask questions via text or voice input, and the app will provide answers based on the content of the uploaded PDFs.

## Features

- **PDF Upload**: Upload multiple PDF files for processing.
- **Text Extraction**: Extract text from uploaded PDFs.
- **Text Chunking**: Split the extracted text into manageable chunks.
- **Vector Store**: Create a vector store using Google's Generative AI embeddings.
- **Chat Interface**: Ask questions via text or voice input.
- **Chat History**: View the history of questions and answers.

## Prerequisites

- Python 3.8 or higher
- Streamlit
- SpeechRecognition
- PyPDF2
- LangChain
- Google Generative AI
- FAISS
- python-dotenv

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/adarshukla3005/Multiple-PDF-Chatbot.git
   cd Multiple-PDF-Chatbot
2. Create a virtual environment:
   ```bash
   Copy
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install the required packages:
   ```bash
   Copy
   pip install -r requirements.txt
   Set up environment variables:

4. Create a .env file in the root directory.

5. Add your Google API key to the .env file:
   plaintext
   Copy
   GOOGLE_API_KEY=your_google_api_key_here
   Usage
6. Run the Streamlit app:
   ```bash
   Copy
   streamlit run app.py
   Upload PDF files:

### Use the sidebar to upload one or more PDF files.

Click the "Submit & Process" button to extract and process the text.

## Ask questions:

Enter your question in the text input box or use the microphone button to ask via voice.

The app will display the answer based on the content of the uploaded PDFs.

## View chat history:

The chat history is displayed below the input box, showing all previous questions and answers.

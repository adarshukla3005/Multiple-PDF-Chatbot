import os
import streamlit as st
import speech_recognition as sr
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set Streamlit page config
st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout="wide")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

# Function to handle voice input
def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak Now 🎙️")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            st.session_state.query_input = text  # Update session state dynamically
        except sr.UnknownValueError:
            st.session_state.query_input = "Could not understand the audio."
        except sr.RequestError:
            st.session_state.query_input = "Could not request results; check your internet connection."
        except sr.WaitTimeoutError:
            st.session_state.query_input = "Listening timed out. Please try again."

# Function to update query input field
def update_query():
    st.session_state.query_input = st.session_state.new_query

# Main UI
def main():
    st.header("Multiple PDF Chatbot 💁")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done ✅")

    # Query input field and mic button in a single row
    col1, col2 = st.columns([8, 1])  # Adjust column widths for better alignment

    with col1:
        st.text_input("Ask a Question", key="new_query", value=st.session_state.query_input, on_change=update_query)

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Adjusts vertical alignment
        if st.button("🎙️", help="Click to record your voice"):
            record_audio()

    # Process query if available
    if st.session_state.query_input:
        reply = user_input(st.session_state.query_input)
        st.write("Reply:", reply)

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Chat History:")
        for chat in st.session_state.chat_history:
            with st.expander(f"Q: {chat['question']}"):
                st.write(f"**A:** {chat['answer']}")

if __name__ == "__main__":
    main()
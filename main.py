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

# Initialize session state for query history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the context, say:
    "Answer is not available in the context." Do not provide incorrect answers.

    Context:
    {context}?

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    reply = response["output_text"]

    # Store query and response in session state
    st.session_state.chat_history.append({"question": user_question, "answer": reply})

    return reply

def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak Now 🎙️")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand the audio."
        except sr.RequestError:
            return "Could not request results; check your internet connection."
        except sr.WaitTimeoutError:
            return "Listening timed out. Please try again."

def update_query_input():
    """
    Callback function to update the query input from voice input.
    """
    voice_text = record_audio()
    if voice_text and "Could not" not in voice_text:
        st.session_state.query_input = voice_text  # Update query input in session state

def main():
    st.header("Multiple PDF Chatbot 🙋🏻‍♂️")

    # Sidebar for file upload and processing
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files 📄", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done ✅")

    # Create columns for better UI alignment
    col1, col2 = st.columns([5, 1])

    with col1:
        # Use a unique key for the text input widget
        query_text = st.text_input(
            "Ask a Question from the PDF Files",
            value=st.session_state.query_input,
            key="query_input_widget"  # Unique key for the widget
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎙️", help="Click to record your voice"):
            update_query_input()  # Call the function to update query input
            st.rerun()  # Force a rerun to reflect the updated query input

    # Process query if available
    if query_text:
        reply = user_input(query_text)
        st.write("Reply:", reply)

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Chat History:")
        for chat in st.session_state.chat_history:
            with st.expander(f"Q: {chat['question']}"):
                st.write(f"**A:** {chat['answer']}")

if __name__ == "__main__":
    main()
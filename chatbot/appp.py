import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from docx import Document
import re
from datetime import datetime, timedelta
import dateparser
import json

# CSS
st.markdown(
    """
     <style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(to bottom, #000000, #1a472a);
        color: #ffffff;  /* White text */
        font-weight: bold;
    }

    /* Sidebar background */
    .css-1d391kg {
        background: linear-gradient(to bottom, #000000, #2a623d);
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        font-weight: bold;
    }

    /* Text input and text area */
    .stTextInput input, .stTextArea textarea {
        background: #2a623d;
        color: #ffffff;
        border: 1px solid #ffffff;
        font-weight: bold;
    }

    /* Buttons */
    .stButton button {
        background-color: #c5d86d;  /* Yellow-Green */
        color: #000000;
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: bold;
        border: none;
    }

    /* Buttons on hover */
    .stButton button:hover {
        background-color: #b0c95f;
    }
</style>


    """,
    unsafe_allow_html=True
)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# API Configuration
if not GOOGLE_API_KEY or not HUGGINGFACEHUB_API_TOKEN:
    st.error("API keys are missing. Please configure them in the .env file.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "user_info" not in st.session_state:
    st.session_state.user_info = {}
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "appointment_details" not in st.session_state:
    st.session_state.appointment_details = {}
if "show_appointment_form" not in st.session_state:
    st.session_state.show_appointment_form = False

# Function to extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    texts = [page.extract_text() for page in reader.pages if page.extract_text()]
    return "\n".join(texts)

# Function to extract text from Word document
def extract_text_from_docx(file):
    doc = Document(file)
    texts = [para.text for para in doc.paragraphs if para.text]
    return "\n".join(texts)

# Function to extract text from text file
def extract_text_from_txt(file):
    return file.read().decode("utf-8")

# Load document and create knowledge base
def load_document_to_knowledge_base(file):
    file_extension = file.name.split(".")[-1].lower()
    if file_extension == "pdf":
        text = extract_text_from_pdf(file)
    elif file_extension == "docx":
        text = extract_text_from_docx(file)
    elif file_extension == "txt":
        text = extract_text_from_txt(file)
    else:
        st.error(f"Unsupported file type: {file_extension}")
        return

    if text:
        st.session_state.vectorstore = FAISS.from_texts([text], embedding=embedding_model)
        st.success(f"{file.name} uploaded and knowledge base created!")

# Retrieve documents
def retrieve_documents(query, k=2):
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)

# Generate response with memory
def generate_response(query, context=""):
    conversation_history = "\n".join(
        [f"User: {msg['query']}\nAI: {msg['response']}" for msg in st.session_state.conversation_history[-5:]]
    )
    
    prompt = f"""
    You are a helpful AI assistant. Use the following conversation history and context if available to answer the user's question:
    
    Conversation History:
    {conversation_history}
    
    Context: {context}
    
    Question: {query}
    Answer:
    """
    response = model.generate_content(prompt)
    return response.text

# Validate Email & Phone Number
def validate_email(email):
    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(pattern, email) is not None

def validate_phone(phone):
    pattern = r'^\+?[0-9]{7,15}$'
    return re.match(pattern, phone) is not None

# Updated date extraction function
def extract_and_validate_date(date_str):
   
    today = datetime.today()
    date = dateparser.parse(
        date_str,
        settings={
            'PREFER_DATES_FROM': 'future',
            'RELATIVE_BASE': today
        }
    )

    if not date:
        date = handle_common_phrases(date_str, today)

    if date:
        return date.strftime("%Y-%m-%d")
    else:
        return None

def handle_common_phrases(date_str, today):
    """
    Handles common phrases like 'next Monday' or 'this Friday' manually if dateparser fails.
    """
    weekdays = [
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
    ]
    date_str = date_str.lower().strip()
    if "next" in date_str or "this" in date_str:
        for i, day in enumerate(weekdays):
            if day in date_str:
                current_weekday = today.weekday()
                target_weekday = i

                if "next" in date_str:
                    days_ahead = (target_weekday - current_weekday + 7) % 7
                    days_ahead = days_ahead if days_ahead != 0 else 7
                elif "this" in date_str:
                    days_ahead = (target_weekday - current_weekday + 7) % 7

                target_date = today + timedelta(days=days_ahead)
                return target_date
    return None

# Save user info locally
def save_user_info_locally(user_info):
    with open("user_info.json", "w") as file:
        json.dump(user_info, file)

# Save appointment details locally
def save_appointment_details_locally(appointment_details):
    with open("appointment_details.json", "w") as file:
        json.dump(appointment_details, file)

# Reset everything
def reset():
    st.session_state.vectorstore = None
    st.session_state.conversation_history.clear()
    st.session_state.user_info.clear()
    st.session_state.form_submitted = False
    st.session_state.appointment_details.clear()
    st.session_state.show_appointment_form = False
    st.rerun()

# UI Components
st.title("GUFFADI AI")

#Show the form first
if not st.session_state.form_submitted:
    st.subheader("Please provide your information for having guff")
    with st.form("user_info_form"):
        name = st.text_input("Enter your Name")
        phone = st.text_input("Enter your Phone Number")
        email = st.text_input("Enter your Email")

        submitted = st.form_submit_button("Submit")
        if submitted:
            if not validate_email(email):
                st.error("Invalid email format. Please enter a valid email.")
            elif not validate_phone(phone):
                st.error("Invalid phone number. Please enter a valid phone number.")
            else:
                st.session_state.user_info = {"name": name, "phone": phone, "email": email}
                save_user_info_locally(st.session_state.user_info)
                st.session_state.form_submitted = True
                st.rerun()  

#Start the chatbot after form submission
if st.session_state.form_submitted:
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Reset"):
            reset()
    with col2:
        if st.button("Make Appointment"):
            st.session_state.show_appointment_form = True

    # Appointment form popup
    if st.session_state.show_appointment_form:
        st.subheader("Book an Appointment")
        with st.form("appointment_form"):
            appointment_date = st.text_input("Enter Appointment Date")
            appointment_phone = st.text_input("Enter Appointment Phone Number", value=st.session_state.user_info.get("phone", ""))
            appointment_email = st.text_input("Enter Appointment Email", value=st.session_state.user_info.get("email", ""))
            confirm = st.form_submit_button("Confirm Appointment")
            
            if confirm:
                formatted_date = extract_and_validate_date(appointment_date)
                if not formatted_date:
                    st.error("Unable to understand the date. Please try again.")
                else:
                    appointment_details = {
                        "date": formatted_date,
                        "phone": appointment_phone,
                        "email": appointment_email
                    }
                    save_appointment_details_locally(appointment_details)
                    st.success(f"Appointment booked for {formatted_date}")
                    st.session_state.show_appointment_form = False
                    st.rerun()

    # Upload document
    uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])
    if uploaded_file:
        load_document_to_knowledge_base(uploaded_file)

    # Display conversation history
    for entry in st.session_state.conversation_history:
        st.markdown(f"**You:** {entry['query']}")
        st.markdown(f"**Chatbot:** {entry['response']}")
        st.markdown("---")

    # Chatbot function
    def chatbot(user_query):
        user_query_lower = user_query.lower()

        if "appointment details" in user_query_lower:
            try:
                with open("appointment_details.json", "r") as file:
                    appointment_details = json.load(file)
                if appointment_details:
                    return (
                        f"Your appointment is scheduled for {appointment_details['date']}.\n"
                        f"Phone: {appointment_details['phone']}\nEmail: {appointment_details['email']}"
                    )
                else:
                    return "No appointment found."
            except FileNotFoundError:
                return "No appointment found."

        elif user_query_lower in ["what is my name?", "what's my name?", "tell me my name"]:
            return f"Your name is {st.session_state.user_info.get('name', 'not provided')}."
        
        elif user_query_lower in ["what is my phone number?", "what's my phone number?", "tell me my phone number"]:
            return f"Your phone number is {st.session_state.user_info.get('phone', 'not provided')}."
        
        elif user_query_lower in ["what is my email?", "what's my email?", "tell me my email"]:
            return f"Your email is {st.session_state.user_info.get('email', 'not provided')}."
        
        elif "book appointment" in user_query_lower:
            date = extract_and_validate_date(user_query)
            if date:
                st.session_state.appointment_details["date"] = date
                save_appointment_details_locally(st.session_state.appointment_details)
                return f"Appointment booked for {date}. Please provide additional details if needed."
            else:
                return "Could not understand the date. Please provide a valid date."
        
        elif st.session_state.vectorstore:
            relevant_docs = retrieve_documents(user_query)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            return generate_response(user_query, context)
        
        else:
            return generate_response(user_query)

    # User input query
    query = st.text_input("Guff garum")
    if st.button("Guff"):
        response = chatbot(query)
        st.session_state.conversation_history.append({"query": query, "response": response})
        st.markdown(f"**Guffadi:** {response}")

    # Display appointment details
    if st.session_state.appointment_details:
        st.subheader("Appointment Details")
        st.write(st.session_state.appointment_details)
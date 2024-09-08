import streamlit as st
import tempfile
from ai_agent import AIAgent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config("Zania")
st.title("Zania, Inc")

st.sidebar.header(" Pdf QA to Slack")
status = st.empty()
file = st.file_uploader("Upload your file")
if file:
    status.success("Successfully uploaded your file")
    with st.spinner("Processing the file"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.getvalue())
            temp_file_path = temp_file.name
        ai_agent = AIAgent(temp_file_path)
    status.write("Enter your questions")
    q = st.text_area("Enter your questions separated by ,")
    if q:
        questions = [i.strip() for i in q.split(",")]
        with st.spinner("Extracting context and generating answers"):
            result = ai_agent.run(questions)
            st.json(result)
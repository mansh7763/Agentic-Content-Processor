import streamlit as st
import requests

st.set_page_config(
    page_title="Agentic Content Processor",
    page_icon="",
    layout="wide"
)

# API endpoint
API_URL = "http://localhost:8000"

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'awaiting_followup' not in st.session_state:
    st.session_state.awaiting_followup = False


def display_result(result: dict, task: str):
    if not result:
        st.warning("No result returned")
        return
    if not result.get("success", True):
        st.error(f"Error: {result.get('error', 'Unknown error')}")
        return

    if task == "summarize":
        st.subheader("Summary")
        st.markdown("**One-Liner:**")
        st.info(result.get("one_liner", ""))
        st.markdown("**Key Points:**")
        for i, bullet in enumerate(result.get("bullets", []), 1):
            st.markdown(f"{i}. {bullet}")
        st.markdown("**Detailed Summary:**")
        st.write(result.get("five_sentences", ""))
    elif task == "sentiment":
        st.subheader("Sentiment analysis")
        label = result.get("label", "neutral")
        confidence = result.get("confidence", 0)
        st.markdown(f"**Sentiment:** {label.upper()}")
        st.progress(confidence)
        st.caption(f"Confidence: {confidence:.0%}")
        st.markdown("**Justification:**")
        st.write(result.get("justification", ""))
    elif task == "code_explain":
        st.subheader("Code explanation")
        if result.get("language"):
            st.badge(f"Language: {result['language']}")
        st.markdown("**Explanation:**")
        st.write(result.get("explanation", ""))
        st.markdown("**Bugs & Issues:**")
        for bug in result.get("bugs", []):
            st.warning(bug)
        st.markdown("**Complexity:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Time", result.get("time_complexity", "N/A"))
        with col2:
            st.metric("Space", result.get("space_complexity", "N/A"))
    elif task == "extract":
        st.subheader("Extracted information")
        action_items = result.get("action_items", [])
        if action_items:
            st.markdown(f"**Found {len(action_items)} action items:**")
            for i, item in enumerate(action_items, 1):
                st.markdown(f"{i}. {item}")
        else:
            st.info("No specific action items found")
    elif task == "qa":
        st.subheader("Answer")
        st.write(result.get("answer", ""))
    else:
        st.json(result)


def handle_followup(response_text: str):
    with st.spinner("Processing your response..."):
        try:
            response = requests.post(
                f"{API_URL}/followup",
                json={
                    "session_id": st.session_state.session_id,
                    "response": response_text
                }
            )
            if response.status_code == 200:
                data = response.json()
                st.session_state.messages.append({
                    "role": "user",
                    "content": response_text
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Completed.",
                    "result": data["result"],
                    "task": data.get("task")
                })
                st.session_state.awaiting_followup = False
                st.session_state.session_id = None
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
    st.rerun()


def process_input(text: str, file):
    if st.session_state.awaiting_followup:
        if text:
            handle_followup(text)
        return
    if text:
        st.session_state.messages.append({
            "role": "user",
            "content": text
        })
    elif file:
        st.session_state.messages.append({
            "role": "user",
            "content": f"Uploaded file: {file.name}"
        })

    with st.spinner("Processing..."):
        try:
            if file:
                files = {"file": (file.name, file.getvalue(), file.type)}
                response = requests.post(f"{API_URL}/process/file", files=files)
            else:
                response = requests.post(
                    f"{API_URL}/process/text",
                    json={"text": text}
                )

            if response.status_code == 200:
                data = response.json()
                if data["status"] == "needs_clarification":
                    st.session_state.session_id = data["session_id"]
                    st.session_state.awaiting_followup = True
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data['question'],
                        "extracted_text": data.get("extracted_text", ""),
                        "metadata": data.get("metadata", {})
                    })
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Completed.",
                        "result": data["result"],
                        "task": data.get("task"),
                        "extracted_text": data.get("extracted_text", ""),
                        "metadata": data.get("metadata", {})
                    })
            else:
                st.error(f"Error: {response.text}")

        except Exception as e:
            st.error(f"Error: {str(e)}")

    st.rerun()

st.title("Agentic Content Processor")
st.markdown(
    "Upload a document, image, audio file, or paste text, and the agent will extract the content and help you summarize, analyze, or ask questions."
)

with st.sidebar:
    st.header("Supported tasks")
    st.markdown("""
    - **Summarization** (short and detailed summaries)
    - **Sentiment analysis** (with confidence)
    - **Code explanation** (issues and complexity)
    - **Text extraction** (OCR from images)
    - **YouTube transcripts**
    - **PDF processing**
    - **Audio transcription**
    - **Question answering** (conversational)
    - **Action item extraction**
    """)
    
    st.header("Supported file types")
    st.markdown("""
    - **Images**: JPG, PNG
    - **PDFs**: Text or Scanned
    - **Audio**: MP3, WAV, M4A
    """)
    
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.session_state.awaiting_followup = False
        st.rerun()

st.header("Conversation")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            if "extracted_text" in message:
                with st.expander("Extracted content"):
                    st.text(message["extracted_text"])
            
            if "metadata" in message:
                with st.expander("Metadata"):
                    st.json(message["metadata"])
            
            if "result" in message:
                display_result(message["result"], message.get("task"))
            else:
                st.markdown(message["content"])
        else:
            st.markdown(message["content"])

col1, col2 = st.columns([3, 1])

with col1:
    text_input = st.text_input(
        "Enter text, a question, or a URL:",
        key="text_input",
        placeholder="Type your message or question here..."
    )

with col2:
    uploaded_file = st.file_uploader(
        "Or upload a file",
        type=['jpg', 'jpeg', 'png', 'pdf', 'mp3', 'wav', 'm4a'],
        key="file_upload"
    )

if st.button("Run", type="primary"):
    if text_input or uploaded_file:
        process_input(text_input, uploaded_file)
    else:
        st.warning("Please enter text or upload a file.")


st.markdown("---")
st.caption("Powered by LangGraph, Groq, and open-source models")
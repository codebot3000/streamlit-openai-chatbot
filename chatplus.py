import json
import time
import streamlit as st
try:
    from resources import system_instructions
except ImportError:
    system_instructions = "You are a helpful assistant."  # Fallback default prompt

# Try importing both backends; if a package is missing we'll show a helpful message
try:
    from openai import OpenAI
    _have_openai = True
except Exception:
    OpenAI = None
    _have_openai = False

try:
    from langchain_openai.chat_models import ChatOpenAI
    _have_langchain = True
except Exception:
    ChatOpenAI = None
    _have_langchain = False

# --- App header and image ---
st.markdown(
    """
    <style>
    h1 {
        font-size: 28px !important;
    }
    h2 {
        font-size: 20px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("ChatPlus — choose your chat backend")

# --- Sidebar controls: select backend and model ---
BACKENDS = ["Direct Call (OpenAI client)", "LangChain (ChatOpenAI)"]
# Store backend selection in session state so we can detect changes between reruns
if "backend" not in st.session_state:
    st.session_state["backend"] = BACKENDS[0]
backend = st.sidebar.selectbox("Choose chat backend", BACKENDS, key="backend")

# If the backend changed and the user selected Direct Call, reset its chat history and timing
prev_backend = st.session_state.get("_prev_backend_for_backend_select")
if backend != prev_backend:
    if backend.startswith("Direct Call"):
        st.session_state["direct_messages"] = []
        st.session_state["last_response_time"] = None
        if "messages" in st.session_state:
            st.session_state["messages"] = []
    st.session_state["_prev_backend_for_backend_select"] = backend

# Allowed model options (ChatGPT-5 family + gpt-4o)
MODEL_OPTIONS = ["gpt-4o", "gpt-5-nano", "gpt-5-mini", "gpt-5"]
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = MODEL_OPTIONS[0]

st.sidebar.selectbox(
    "Choose model",
    options=MODEL_OPTIONS,
    index=MODEL_OPTIONS.index(st.session_state["openai_model"]),
    key="openai_model",
)

system_prompt = st.sidebar.text_area("System prompt (optional):", value=system_instructions, height=500)

# Read API key from Streamlit secrets once for both backends
openai_api_key = None
if hasattr(st, "secrets"):
    openai_api_key = st.secrets.get("OPENAI_API_KEY")

# --- Helper: pretty display for model responses ---
def display_readable(raw: object):
    """Render model output in a readable form: JSON, code block, or plain text."""
    if raw is None:
        st.write("(no response)")
        return

    # If it's already a dict/list, show JSON
    if isinstance(raw, (dict, list)):
        st.json(raw)  # type: ignore
        return

    text = str(getattr(raw, "content", raw))
    # try parse JSON
    try:
        parsed = json.loads(text)
        st.json(parsed)
        return
    except Exception:
        pass

    # long or multiline text -> code block for readability
    if "\n" in text or len(text) > 300:
        st.code(text)
    else:
        st.write(text)


# --- Direct Call handler (uses OpenAI client streaming) ---
def call_direct(prompt: str):
    """Send prompt to OpenAI client with streaming and store messages in session state."""
    if not _have_openai:
        st.error("openai package not installed. Install with: pip install openai")
        return
    if not openai_api_key:
        st.warning("Set OPENAI_API_KEY in Streamlit secrets to use Direct Call.")
        return

    # Initialize client and message history
    if OpenAI is None:
        st.error("OpenAI client is not available. Please ensure the openai package is installed correctly.")
        return
    client = OpenAI(api_key=openai_api_key)
    if "direct_messages" not in st.session_state:
        st.session_state["direct_messages"] = []

    # Append user message and display
    st.session_state["direct_messages"].append({"role": "user", "content": prompt})
    for message in st.session_state["direct_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Prepare messages to send to the API (include system prompt, but keep it out of displayed history)
    if system_prompt:
        st.session_state["direct_messages"].append({"role": "system", "content": system_prompt})

    # Call the API (streaming) and render response
    try:
        start = time.time()
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state["direct_messages"]],
            stream=True,
        )
        response = st.write_stream(stream)
        elapsed = time.time() - start
        st.session_state["last_response_time"] = elapsed
    except Exception as e:
        st.error(f"Error calling OpenAI client: {e}")
        st.session_state["last_response_time"] = None
        return

    # Save assistant reply
    st.session_state["direct_messages"].append({"role": "assistant", "content": response})


# --- LangChain handler (ChatOpenAI) ---
def call_langchain(prompt: str):
    """Send prompt to the LangChain ChatOpenAI wrapper and display formatted output."""
    if not _have_langchain:
        st.error("langchain-openai package not installed. Install with: pip install langchain-openai")
        return
    if not openai_api_key:
        st.warning("Set OPENAI_API_KEY in Streamlit secrets to use LangChain backend.")
        return

    # Construct ChatOpenAI client; handle variations in kwarg names
    if ChatOpenAI is None:
        st.error("ChatOpenAI class not available. Check your langchain-openai installation.")
        return
    try:
        model = ChatOpenAI(model=st.session_state["openai_model"], temperature=0.7, api_key=openai_api_key)
    except TypeError:
        model = ChatOpenAI(model=st.session_state["openai_model"], temperature=0.7, api_key=openai_api_key)

    try:
        start = time.time()
        # Combine system prompt (if provided) with the user prompt so ChatOpenAI receives the system instructions
        prompt_to_send = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        raw = model.invoke(prompt_to_send)
        elapsed = time.time() - start
        st.session_state["last_response_time"] = elapsed
    except Exception as e:
        st.error(f"Error calling ChatOpenAI: {e}")
        st.session_state["last_response_time"] = None
        return

    display_readable(raw)


# --- Main input area shared between backends ---
with st.form("chat_form"):
    user_input = st.text_area("Enter your prompt:", "What are the three key pieces of advice for learning how to code?", height=120)
    submitted = st.form_submit_button("Send")

    if submitted and user_input:
        # Route the prompt to the selected backend
        if backend.startswith("Direct Call"):
            call_direct(user_input)
        else:
            call_langchain(user_input)

        # Small note for the user about chosen backend/model and response time
        rt = st.session_state.get("last_response_time")
        if rt is None:
            rt_text = "(no timing)"
        else:
            rt_text = f"{rt:.2f} s"
        st.caption(f"Used backend: {backend} — model: {st.session_state['openai_model']} — response time: {rt_text}")

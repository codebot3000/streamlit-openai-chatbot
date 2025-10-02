# Introduction 
Streamlit test application for testing OpenAI LLM calls using chat completions or LangChain and providing a system prompt. 
Requires OpenAI API Platform account and API key as well as the OpenAI Python Library.

## Install Dependencies

Install the required Python dependencies.
```powershell
pip install openai streamlit
```

## Add OpenAI key to Streamlit secrets
[Streamlit Secrets Management](https://docs.streamlit.io/develop/concepts/connections/secrets-management)

```python
# .streamlit/secrets.toml
OPENAI_API_KEY = "YOUR_API_KEY"
```

# Getting Started
Clone code locally.
```powershell
git clone https://github.com/codebot3000/streamlit-openai-chatbot.git
```

# Build and Test
Run application locally using:
```powershell
# PowerShell
streamlit run chatplus.py
```

# Resources
https://docs.streamlit.io/

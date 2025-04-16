# Groq AI Chatbot

A simple chatbot powered by Groq's LLama3 model, built with Streamlit. This chatbot allows you to have interactive conversations with Groq's state-of-the-art LLM.

## Features

- Interactive chat interface with Streamlit
- Powered by Groq's LLama3-8B model
- Conversation history tracking
- Secure API key handling
- Dark theme interface

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/chathurab1120/LangGraph_Agentic_AI_Chatbot.git
   cd LangGraph_Agentic_AI_Chatbot
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   
   # On Windows
   .\venv\Scripts\activate
   
   # On Mac/Linux
   source venv/bin/activate
   
   # Install required packages
   pip install -r requirements.txt
   ```

3. Create a `.env` file for your API key:
   ```
   GROK_API_KEY=your_groq_api_key_here
   ```

## Running Locally

1. Make sure your virtual environment is activated:
   ```
   # On Windows
   .\venv\Scripts\activate
   
   # On Mac/Linux
   source venv/bin/activate
   ```

2. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

3. The app will be available at http://localhost:8501

## Deployment to Streamlit Cloud

### Preparing Your Repository

1. Make sure your code is pushed to GitHub:
   ```
   git add .
   git commit -m "Prepare for Streamlit deployment"
   git push
   ```

2. Ensure that your `.env` file is in `.gitignore` to keep your API key secure.

### Setting Up on Streamlit Cloud

1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository, branch, and the `streamlit_app.py` file
5. In the "Advanced settings":
   - Add your API key as a secret using TOML format:
     ```toml
     GROK_API_KEY = "your_groq_api_key_here"
     ```

6. Click "Deploy!"
7. Your app will be deployed and accessible via a Streamlit URL

## Dark Theme Configuration

The app is configured with a dark theme by default. This is controlled by the `.streamlit/config.toml` file with the following settings:

```toml
[theme]
base = "dark"
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"
```

## Security Notes

- Never commit your API key to the repository
- Use environment variables or Streamlit secrets for sensitive information
- For production, consider setting up proper authentication

## Project Structure

- `streamlit_app.py` - Main Streamlit application
- `.streamlit/config.toml` - Theme configuration
- `requirements.txt` - Python dependencies
- `.env` - Local environment variables (not in GitHub)
- `.gitignore` - Files to exclude from GitHub

## License

[MIT License](LICENSE) 
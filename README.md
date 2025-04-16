# Groq AI Chatbot

A simple chatbot powered by Groq's LLama3 model, built with Streamlit.

## Features

- Interactive chat interface
- Powered by Groq's LLama3-8B model
- Conversation history tracking
- Secure API key handling

## Installation

1. Clone this repository:
   ```
   git clone <your-repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Create a `.env` file for your API key:
   ```
   GROK_API_KEY=your_groq_api_key_here
   ```

## Running Locally

Run the Streamlit app:
```
streamlit run streamlit_app.py
```

The app will be available at http://localhost:8501

## Deployment to Streamlit Cloud

To deploy to Streamlit Cloud:

1. Push your code to GitHub (make sure `.env` is in `.gitignore`)
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository, branch, and the `streamlit_app.py` file
6. In the "Advanced settings", add your secrets:
   - Add `GROK_API_KEY` and its value

Your app will be deployed and accessible via a Streamlit URL.

## Security Notes

- Never commit your API key to the repository
- Use environment variables or Streamlit secrets for sensitive information
- For production, consider setting up proper authentication

## License

[Specify your license here] 
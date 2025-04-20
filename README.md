# LangGraph Agentic AI Chatbot

An advanced chatbot implementation using LangGraph for structured conversations and tool usage. The chatbot can access multiple external tools including Wikipedia, Arxiv, and Tavily Search to provide comprehensive responses.

## Features

- Multi-tool integration (Wikipedia, Arxiv, Tavily Search)
- Structured conversation flow using LangGraph
- Intelligent tool selection based on query context
- Comprehensive error handling
- 100% test coverage
- Modern dependency management

## Prerequisites

- Python 3.8+
- Groq API key
- Tavily API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LangGraph_Agentic_AI_Chatbot_REACT_Structure.git
cd LangGraph_Agentic_AI_Chatbot_REACT_Structure
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a .env file with your API keys:
```
GROK_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Running the Chatbot

1. Start the chatbot:
```bash
python enhanced_chatbot.py
```

## Development

To set up the development environment:

1. Install development dependencies:
```bash
pip install -r test-requirements.txt
```

2. Run tests:
```bash
python run_tests.py
```

## Project Structure

- `enhanced_chatbot.py`: Main chatbot implementation
- `test_enhanced_chatbot.py`: Test suite
- `requirements.txt`: Production dependencies
- `test-requirements.txt`: Development dependencies
- `run_tests.py`: Test runner with coverage reporting

## Testing

The project maintains 100% test coverage and includes tests for:
- Message conversion
- Agent creation
- Direct responses
- Tool usage
- Error handling

## License

MIT License 
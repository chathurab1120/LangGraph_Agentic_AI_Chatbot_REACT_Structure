import unittest
from unittest.mock import patch, MagicMock
import os
import json
from enhanced_chatbot import (
    convert_to_langchain_messages,
    create_agent,
    process_message,
    ChatState
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

class TestEnhancedChatbot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment variables"""
        os.environ["GROK_API_KEY"] = "test_grok_key"
        os.environ["TAVILY_API_KEY"] = "test_tavily_key"

    def setUp(self):
        """Set up test fixtures before each test method"""
        self.test_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

    def test_convert_to_langchain_messages(self):
        """Test conversion of dict messages to LangChain message objects"""
        langchain_msgs = convert_to_langchain_messages(self.test_messages)
        
        # First message should be system prompt
        self.assertIsInstance(langchain_msgs[0], SystemMessage)
        
        # Test user message conversion
        self.assertIsInstance(langchain_msgs[1], HumanMessage)
        self.assertEqual(langchain_msgs[1].content, "Hello")
        
        # Test assistant message conversion
        self.assertIsInstance(langchain_msgs[2], AIMessage)
        self.assertEqual(langchain_msgs[2].content, "Hi there!")

    @patch('enhanced_chatbot.ChatGroq')
    def test_create_agent(self, mock_chat_groq):
        """Test agent creation and configuration"""
        # Mock the LLM
        mock_llm = MagicMock()
        mock_chat_groq.return_value = mock_llm
        
        # Create agent
        chain = create_agent()
        
        # Verify LLM initialization
        mock_chat_groq.assert_called_once_with(
            api_key="test_grok_key",
            model_name="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=4096
        )
        
        # Verify chain creation
        self.assertIsNotNone(chain)

    @patch('enhanced_chatbot.ChatGroq')
    def test_process_message_direct_response(self, mock_chat_groq):
        """Test processing a message that doesn't require tools"""
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = "This is a direct response"
        mock_response.additional_kwargs = {}
        
        mock_llm = MagicMock()
        mock_llm.predict_messages.return_value = mock_response
        mock_chat_groq.return_value = mock_llm
        
        chain = create_agent()
        result = process_message(chain, "Hello", [])
        
        # Verify the response structure
        self.assertIsInstance(result, list)
        self.assertEqual(result[-1]["role"], "assistant")
        self.assertEqual(result[-1]["content"], "This is a direct response")

    @patch('enhanced_chatbot.ChatGroq')
    def test_process_message_with_tool(self, mock_chat_groq):
        """Test processing a message that requires tool usage"""
        # Mock LLM responses for tool selection and final response
        mock_tool_selection = MagicMock()
        mock_tool_selection.additional_kwargs = {
            "function_call": {
                "name": "wikipedia",
                "arguments": json.dumps({"query": "test query"})
            }
        }
        
        mock_final_response = MagicMock()
        mock_final_response.content = "Response after using Wikipedia"
        mock_final_response.additional_kwargs = {}
        
        mock_llm = MagicMock()
        mock_llm.predict_messages.side_effect = [
            mock_tool_selection,
            mock_final_response
        ]
        mock_chat_groq.return_value = mock_llm
        
        chain = create_agent()
        result = process_message(chain, "What is quantum computing?", [])
        
        # Verify tool usage and response
        self.assertIsInstance(result, list)
        self.assertTrue(any("Tool wikipedia returned" in msg["content"] 
                          for msg in result if msg["role"] == "system"))

    def test_error_handling(self):
        """Test error handling in message processing"""
        with patch('enhanced_chatbot.ChatGroq') as mock_chat_groq:
            # Make the LLM raise an exception
            mock_llm = MagicMock()
            mock_llm.predict_messages.side_effect = Exception("Test error")
            mock_chat_groq.return_value = mock_llm
            
            chain = create_agent()
            result = process_message(chain, "This should fail", [])
            
            # Verify error response
            self.assertEqual(result[-1]["role"], "assistant")
            self.assertTrue("error" in result[-1]["content"].lower())

    def test_process_message_chain_error(self):
        """Test error handling when chain.invoke fails"""
        with patch('enhanced_chatbot.ChatGroq') as mock_chat_groq:
            # Mock the chain to raise an exception
            mock_llm = MagicMock()
            mock_llm.predict_messages.side_effect = Exception("Chain error")
            mock_chat_groq.return_value = mock_llm
            
            chain = MagicMock()
            chain.invoke.side_effect = Exception("Chain invocation error")
            
            history = [{"role": "user", "content": "Previous message"}]
            result = process_message(chain, "This should trigger chain error", history)
            
            # Verify the error response
            self.assertEqual(len(result), 3)  # Original history + new message + error
            self.assertEqual(result[-2]["role"], "user")
            self.assertEqual(result[-2]["content"], "This should trigger chain error")
            self.assertEqual(result[-1]["role"], "assistant")
            self.assertTrue("error" in result[-1]["content"].lower())
            self.assertTrue("try again" in result[-1]["content"].lower())

if __name__ == '__main__':
    unittest.main() 
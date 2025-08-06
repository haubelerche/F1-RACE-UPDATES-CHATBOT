"""
F1 RAG Chatbot Streamlit UI
Simple web interface for the F1 chatbot
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import from app
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

from app.main import create_chatbot

# Page configuration
st.set_page_config(
    page_title="F1 RAG Chatbot",
    page_icon="🏎️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize chatbot
@st.cache_resource
def get_chatbot():
    """Initialize and cache the chatbot"""
    data_path = os.path.join(parent_dir, "data")
    return create_chatbot(data_path=data_path)

def main():
    st.title("🏎️ F1 RAG Chatbot")
    st.markdown("Ask me anything about Formula 1 racing!")
    
    # Initialize chatbot
    try:
        chatbot = get_chatbot()
        
        # Show system info in sidebar
        with st.sidebar:
            st.header("System Info")
            if st.button("Refresh Info"):
                st.cache_resource.clear()
                st.rerun()
            
            info = chatbot.get_system_info()
            st.write(f"**Device:** {info['model_device']}")
            st.write(f"**Documents:** {info['documents_loaded']}")
            st.write(f"**Status:** {'✅ Ready' if info['initialized'] else '⏳ Loading...'}")
        
        # Initialize session state for chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "Hello! I'm your F1 expert assistant. Ask me anything about Formula 1 racing, recent news, drivers, or race results!"
            })
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about F1..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get chatbot response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = chatbot.chat(prompt)
                        st.markdown(response)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Clear chat button
        if st.sidebar.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
            
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {str(e)}")
        st.info("Make sure you have run the data preparation pipeline first!")

if __name__ == "__main__":
    main()
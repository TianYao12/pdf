import streamlit as st
from dotenv import load_dotenv
# from langchain_community.chat_models import ChatOpenAI
from breakdown import get_pdf_text, get_text_chunks, get_vectorstore
from htmlTemplates import css, bot_template, user_template
from memory import get_conversation_chain

# robot and human responses shown on streamlit
def handle_userinput(user_question):
    response = st.session_state.conversation(
        {'question': user_question})  # remembers previous context
    # st.session_state returns the chat data in JSON? format
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Chatbot")
    st.write(css, unsafe_allow_html=True)
    # st.write
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("PDF Chatbot")
    user_question = st.text_input("Ask a question about your PDF document: ")
    if user_question:
        handle_userinput(user_question)

    # allow streamlit to parse html
    # st.write(user_template.replace(
    #     "{{MSG}}", "hello human"), unsafe_allow_html=True)
    # st.write(bot_template.replace(
    #     "{{MSG}}", "hello robot"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_documents = st.file_uploader(
            "Upload your PDF documents", accept_multiple_files=True)
        if st.button("Process"):
            # user sees spinner while the following code runs
            with st.spinner("Processing..."):
                # get pdf raw text
                raw_text = get_pdf_text(pdf_documents)
                st.write(raw_text)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                # variable is linked to session state instead of reinitialized
                # (streamlit sometimes reruns entire code whenever changes occur on user's end)
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

if __name__ == '__main__':
    main()

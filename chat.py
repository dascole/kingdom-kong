import streamlit as st
import os, time
import logging
import requests
from openai import OpenAI
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 500))
DATA_PATH = os.getenv("DATA_PATH", "~/Documents/data")
GW = os.getenv("GW", "http://localhost")
KONG_ADMIN_PORT = os.getenv("GW_ADMIN_PORT", "18001")
KONG_PROXY_PORT = os.getenv("GW_PROXY_PORT", "18000")
PLUGIN_INSTANCE = os.getenv("PLUGIN_INSTANCE", "ai-rag-plugin")
WORKSPACE = os.getenv("WORKSPACE", "default")
MODEL = os.getenv("MODEL", "gpt-4o")
AI_HOST = os.getenv("AI_HOST", f"{GW}:{KONG_PROXY_PORT}/rag")
API_KEY = os.getenv("API_KEY", "xyz")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_text(documents, chunk_size, chunk_overlap):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        logger.error(f"Failed to split documents: {e}")
        st.error("Error while splitting text into chunks.")
        return []


def load_markdown_files(directory_path):
    try:
        loader = DirectoryLoader(
            directory_path,
            glob="*.md",
            loader_cls=UnstructuredMarkdownLoader,
            recursive=True
        )
        return loader.load()
    except Exception as e:
        logger.error(f"Error loading markdown files from {directory_path}: {e}")
        st.error("Failed to load markdown files.")
        return []


def post_chunks_to_api(chunk,url):
    try:
        chunk_content = chunk.page_content
        resp = requests.post(
            url,
            json={"content": chunk_content},
            timeout=10
        )

        if resp.status_code == 200:
            logger.info("Chunk posted successfully.")
        else:
            logger.warning(f"Failed to post chunk. Status: {resp.status_code}, Response: {resp.text}")
            st.warning(f"Chunk post failed: {resp.status_code}")
    except requests.RequestException as e:
        logger.error(f"Request failed when posting chunk: {e}")
        st.error("Network error while posting chunk to API.")

def get_plugin_id(instance: str):
    try:
        resp = requests.get(f"{GW}:{KONG_ADMIN_PORT}/plugins/{PLUGIN_INSTANCE}", timeout=10) 
    except requests.RequestException as e:
        logger.error(f"Request failed when getting plugin ID: {e}")
        st.error("Network error while getting plugin ID.")
        return None

    if resp.status_code == 200:
        return resp.json().get("id")
    else:
        logger.error(f"Failed to get plugin ID. Status: {resp.status_code}, Response: {resp.text}")
        st.error("Failed to get plugin ID from the API.")
        return None
    
def main():
    # Set our page title and icon
    st.set_page_config(page_title="Kong AI Gateway", page_icon=":guardsman:", layout="wide")
    st.title("Kong AI Gateway")   

    # Setup our OAI client
    try:
        client = OpenAI(
            api_key=API_KEY,
            base_url=AI_HOST
        )
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        st.error("Failed to connect to the AI backend.")
        return

    # See if we have conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # See if we have ingested data already
    if "ingested" not in st.session_state:
        st.session_state.ingested = False

    if "split" not in st.session_state:
        st.session_state.split = False
        
    # Create the side panel to host the plugin selection and navigation
    with st.sidebar:
        if "sidebar_page" not in st.session_state:
            st.session_state.sidebar_page = 0 
        
        # Lots of options for AI Plugins 
        plugin = st.selectbox("Select a plugin:", [
            "AI RAG Injector"
        ])
        
        
        # Adds the steps for using the specific plugin, i.e RAG Injector  
        # Uses the page number to determine which 'step' we are on   
        if plugin == "AI RAG Injector" and st.session_state.sidebar_page == 0:
            st.markdown("### Step 1: Ingest Data")
            st.markdown("1. Click the `split` tab ", help="this is helpful") 
            st.markdown("2. Choose a `chunk size`, this is the number of characters you want to split the data into.")
            st.markdown("3. Choose a `chunk overlap`, this is the number of characters that overlap between chunks.")
            st.markdown("4. Click `Load and Split Documents` to load the data.")
            st.markdown("**Note:** The ingestion process may take some time depending on the size of the data.")
            st.divider()
        elif plugin == "AI RAG Injector" and st.session_state.sidebar_page == 1: 
            st.markdown("### Step 2: Use our custom data.")
            st.markdown("Now that we have ingested the data, we can use it to answer questions specific to KIC. ")
        elif plugin == "AI RAG Injector" and st.session_state.sidebar_page == 2:
            st.markdown("### Step 3: Ingest Data")
    
        # Setup a 2x2 grid for the back/next buttons
        col1, col2 = st.columns([1, 1])
        
        # Just make sure we don't go too far back or forward
        with col1:
            if st.button("⬅️ Back"):
                if st.session_state.sidebar_page > 0:
                    st.session_state.sidebar_page -= 1

        with col2:
            if st.button("Next ➡️"):
                st.session_state.sidebar_page += 1  
                
    if plugin == "AI RAG Injector":
        split, ingest, chat, debug = st.tabs(["Split","Ingest content", "Chat","Debug"])
        with split:
            user_chunk_size = st.slider("Chunk size (how many characters you want): ", 1, 1000, CHUNK_SIZE)
            st.write(user_chunk_size)
            user_chunk_overlap = st.slider("Chunk overlap: ", 1, 900, CHUNK_OVERLAP)
            st.write(user_chunk_overlap)
            # See if we already split the docs
            if st.session_state.split:
                st.success("Documents loaded and split successfully.")
            else:
                with st.spinner("Loading and splitting documents...", show_time=True):
                    if st.button("Load and Split Documents"):
                        try:
                            documents = load_markdown_files(DATA_PATH)
                            if not documents:
                                st.warning("No documents loaded. Please check the data path or file contents.")
                                chunks = []
                            else:
                                chunks = split_text(documents, user_chunk_size, user_chunk_overlap)
                                total_chunks = len(chunks) 
                                st.session_state.total_chunks = total_chunks 
                                st.session_state.chunks = chunks
                                if not chunks:
                                    st.warning("No text chunks were created from the documents.")
                        except Exception as e:
                            logger.error(f"Failed to load documents: {e}")
                            st.error("Error while loading documents.")
                            chunks = []
                        st.session_state.split = True
                        st.success("Documents splitsuccessfully.")
            # Load docs and chunk them, though we should probably move this to a separate function
            st.markdown(f"Load up some [markdown from](https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.directory.DirectoryLoader.html): \
                \n\n `{DATA_PATH}` and split them into chunks.")
            
        with ingest:
            if 'plugin_id' not in st.session_state:
                plugin_id = get_plugin_id(PLUGIN_INSTANCE)        
                if plugin_id is None:
                  st.error("Failed to get plugin ID. Please check the Kong Gateway.")
                st.session_state.plugin_id = plugin_id
                

            st.markdown("We will use [Langchain text_splitter](https://api.python.langchain.com/en/latest/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html) to split the documents into chunks.")
            with st.expander("Show code example"):
                st.code (f"""text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                add_start_index=True
            """)
            st.header("Ingest Data")
            if st.session_state.get("ingested", False):
                st.success("All chunks posted successfully.")
            else:
                if st.button("Start Ingestion"):
                    progress_text = "Operation in progress. Please wait."
                    chunk_progress = st.progress(0, text=progress_text)
                    total_chunks = st.session_state.get("total_chunks", 0) 
                    chunks = st.session_state.get("chunks", 0)
                    st.markdown(f"Ingesting data, total chunks: {total_chunks}")
                    
                    
                    for i, chunk in enumerate(chunks):
                        url = f"{GW}:{KONG_ADMIN_PORT}/ai-rag-injector/{st.session_state.plugin_id}/ingest_chunk"
                        post_chunks_to_api(chunk,url)
                        time.sleep(0.2) 
                        progress_percentage = int((i + 1) / total_chunks * 100)
                        chunk_progress.progress(progress_percentage, text=f"{progress_percentage}% complete")

                    st.session_state.ingested = True
                    st.success("All chunks posted successfully.")
            st.markdown("This is done by sending a POST request to the Kong Gateway Admin API endpoint with each chunk.")
            plugin_id = st.session_state.get("plugin_id", None)
            st.code(f"{GW}:{KONG_ADMIN_PORT}/ai-rag-injector/{plugin_id}/ingest_chunk")
        with chat:
            if st.session_state.get("ingested", True):
                st.badge("Ingestion complete. You can now ask questions about the ingested data.", icon="🔥",color="green")
            for msg in st.session_state.messages:
                avatar = "🦍" if msg["role"] == "user" else "🤖"
                
                with st.chat_message(msg["role"],avatar=avatar):
                    st.markdown(msg["content"])

            with st.form(key="chat_form", clear_on_submit=True):
                prompt = st.text_input("Enter your message:")
                submit = st.form_submit_button("Send")

                if submit and prompt:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    try:
                        response = client.chat.completions.create(
                            model=MODEL,
                            messages=st.session_state.messages,
                            temperature=0.5
                        )
                        reply = response.choices[0].message.content
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                        st.rerun()
                    except Exception as e:
                        logger.error(f"Error during chat completion: {e}")
                        st.error("Failed to get response from the AI model.")
        with debug:
            st.markdown("### Debug")
    elif plugin == "AI Proxy":
        print("pass")


if __name__ == '__main__':
    main()

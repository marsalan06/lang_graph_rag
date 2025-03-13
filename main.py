import streamlit as st
from pipeline import CRAGPipeline, display_graph
import logging
from datetime import datetime
from vector_store import VectorStore
import os
import uuid
from auth import authenticate_user, register_external_user, save_session_to_firestore, load_session_from_firestore, delete_session_from_firestore

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@st.cache_resource
def load_pipeline():
    crag = CRAGPipeline()
    display_graph(crag)
    return crag

@st.cache_resource
def load_vector_store():
    return VectorStore()

def display_login_register():
    """Displays login or register UI and handles authentication."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())  # Set a unique session ID

    st.title("Welcome to CRAG Chatbot")
    auth_option = st.radio("Choose an option:", ("Login", "Register"), key="auth_option")

    if auth_option == "Login":
        st.subheader("Login")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password (optional for Firebase)", type="password", key="login_password")
        if st.button("Login", key="login_button"):
            user_data, error_message = authenticate_user(email, password or None)
            if user_data:
                st.session_state.user = user_data
                save_session_to_firestore(user_data, st.session_state.session_id)  # Use function from auth.py
                st.success(f"Logged in as {user_data['user_email']}")
                st.rerun()
            else:
                st.error(error_message or "Invalid credentials or authentication failed.")
    
    elif auth_option == "Register":
        st.subheader("Register")
        email = st.text_input("Email", key="register_email")
        password = st.text_input("Password", type="password", key="register_password")
        user_name = st.text_input("Username", key="register_username")
        tenant_id = st.text_input("Tenant ID", key="register_tenant_id")
        if st.button("Register", key="register_button"):
            if not tenant_id:
                st.error("Tenant ID is required for registration.")
            else:
                user_data, error_message = register_external_user(email, password, user_name, tenant_id)
                if user_data:
                    st.session_state.user = user_data
                    save_session_to_firestore(user_data, st.session_state.session_id)  # Use function from auth.py
                    st.success(f"Registered and logged in as {user_data['user_email']}")
                    st.rerun()
                else:
                    st.error(error_message or "Registration failed.")

def main():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())  # Initialize session ID

    # Check for existing session in Firestore
    if "user" not in st.session_state:
        saved_user = load_session_from_firestore(st.session_state.session_id)  # Use function from auth.py
        if saved_user:
            st.session_state.user = saved_user  # Restore user session
        else:
            display_login_register()
            return
        
    crag = load_pipeline()
    vector_store = load_vector_store()

    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'namespace' not in st.session_state:
        st.session_state.namespace = "default"
    if 'metadata_filter' not in st.session_state:
        tenant_id = st.session_state.user.get("tenant_id")
        st.session_state.metadata_filter = {"tenant_id": {"$eq": tenant_id}} if tenant_id else {}

    # Streamlit UI with chatbot theme
    st.title("CRAG Chatbot")
    st.write(f"Welcome, {st.session_state.user['user_email']}! - Tenant {st.session_state.user['tenant_id']} - Chat with me! Ask a question or say something nice. Type 'exit' to stop.")

    # Sidebar for controls
    with st.sidebar:
        st.image("crag_graph.png", caption="CRAG Pipeline Execution Graph", use_container_width=True)

        if st.button("Logout", key="logout_button"):
            delete_session_from_firestore(st.session_state.session_id)  # Use function from auth.py
            del st.session_state.user
            st.session_state.history = []
            st.session_state.messages = []
            st.session_state.namespace = "default"
            st.session_state.metadata_filter = {}
            st.success("You have been logged out.")
            st.rerun()


        # Get index stats only once and store in session state
        if "index_stats" not in st.session_state:
            st.session_state.index_stats = vector_store.describe_index_stats()
        namespaces = list(st.session_state.index_stats["namespaces"].keys()) if st.session_state.index_stats and "namespaces" in st.session_state.index_stats else ["default"]
        selected_namespace = st.selectbox(
            "Select Namespace",
            options=namespaces,
            index=namespaces.index(st.session_state.namespace) if st.session_state.namespace in namespaces else 0,
            key="namespace_select"
        )

        # Always show source/category filters
        metadata_keys = ["source", "category"]
        selected_key = st.selectbox("Select Metadata Key", options=metadata_keys, key="metadata_key_select")
        selected_value = st.text_input(f"Enter Value for {selected_key}", key="metadata_value_input")

        # Apply settings without full reload
        if st.button("Apply Settings", key="apply_settings"):
            st.session_state.namespace = selected_namespace
            tenant_id = st.session_state.user.get("tenant_id")
            new_filter = {}
            if tenant_id:
                new_filter["tenant_id"] = {"$eq": tenant_id}
            if selected_value:
                new_filter[selected_key] = {"$eq": selected_value}
            st.session_state.metadata_filter = new_filter
            st.rerun()
        
        # File upload section
        st.subheader("Upload PDF Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_upload")
        upload_namespace = st.text_input("Namespace for Upload", value="default", key="upload_namespace")
        upload_source = st.text_input("Source Metadata", value="Unknown", key="upload_source")
        chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=500, step=50, key="chunk_size")
        chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=50, step=10, key="chunk_overlap")

        if st.button("Upload and Index", key="upload_button") and uploaded_file:
            with st.spinner("Processing PDF..."):
                # Save uploaded file temporarily
                temp_file_path = f"temp_{uploaded_file.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Index the document
                try:
                    vector_store.load_and_index_pdf(
                        file_path=temp_file_path,
                        namespace=upload_namespace,
                        source=upload_source,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    st.success(f"Successfully indexed '{uploaded_file.name}' into namespace '{upload_namespace}'!")
                    # Update index stats
                    st.session_state.index_stats = vector_store.describe_index_stats()
                except Exception as e:
                    st.error(f"Failed to index PDF: {e}")

                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

    # Chat area (only updates on new messages)
    chat_container = st.container()
    with chat_container:
        for entry in st.session_state.history:
            with st.chat_message("user"):
                st.write(f"**You ({datetime.now().strftime('%H:%M:%S')})**: {entry['query']}")
            with st.chat_message("assistant"):
                st.write(f"**Bot ({datetime.now().strftime('%H:%M:%S')})**: {entry['response']}")

    # Input area at the bottom
    with st.container():
        user_query = st.chat_input("Type your message here...")
        if user_query:
            current_time = datetime.now().strftime('%H:%M:%S')
            with chat_container.chat_message("user"):
                st.write(f"**You ({current_time})**: {user_query}")
            if user_query.lower() == "exit":
                with chat_container.chat_message("assistant"):
                    st.write(f"**Bot ({current_time})**: Goodbye!")
                st.session_state.history.append({"query": user_query, "response": "Session ended."})
                st.session_state.messages.append({"role": "user", "content": user_query})
                st.session_state.messages.append({"role": "assistant", "content": "Goodbye!"})
                st.stop()
            elif user_query:
                logging.info(f"🎤 User Input: {user_query}")
                with st.spinner("Thinking..."):
                    # Pass the current message history to the pipeline
                    response, updated_messages = crag.run(
                        user_query,
                        namespace=st.session_state.namespace,
                        metadata_filter=st.session_state.metadata_filter,
                        messages=st.session_state.messages
                    )
                with chat_container.chat_message("assistant"):
                    st.write(f"**Bot ({current_time})**: {response}")
                # Update session state with the new history and messages
                st.session_state.history.append({"query": user_query, "response": response})
                st.session_state.messages = updated_messages
                st.rerun()

    # Optional: Clear chat button
    if st.button("Clear Chat", key="clear_chat_button"):
        st.session_state.history = []
        st.session_state.messages = []
        st.rerun()

    if st.checkbox("Show Logs", key="show_logs"):
        log_output = "\n".join([f"{record.asctime} - {record.levelname} - {record.message}"
                                for record in logging.getLogger().handlers[0].buffer])
        st.text_area("Logs", log_output, height=200)

if __name__ == "__main__":
    main()
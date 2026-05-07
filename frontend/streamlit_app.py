import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Production RAG Chatbot",
    page_icon="📚"
)

st.title("📚 Production RAG Chatbot")
st.write("Upload PDFs and ask questions using a FastAPI + LangGraph backend.")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if st.button("Upload and Index"):
    if uploaded_file is None:
        st.warning("Please upload a PDF first.")
    else:
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf"
            )
        }

        response = requests.post(
            f"{API_BASE_URL}/upload",
            files=files
        )

        if response.status_code == 200:
            st.success(response.json())
        else:
            st.error(response.text)


question = st.chat_input("Ask a question from your uploaded PDF")

if question:
    st.chat_message("user").write(question)

    response = requests.post(
        f"{API_BASE_URL}/chat",
        json={"question": question}
    )

    if response.status_code == 200:
        data = response.json()

        st.chat_message("assistant").write(data["answer"])

        with st.expander("Sources"):
            for source in data["sources"]:
                st.write(f"Source: {source.get('source')}")
                st.write(f"Page: {source.get('page')}")
                st.write(source.get("content"))
                st.divider()
    else:
        st.error(response.text)
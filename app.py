

import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

st.set_page_config(
    page_title="Monash Malaysia Academic Advisor",
    page_icon="🎓",
    layout="centered"
)

# Pull API key securely from Streamlit Cloud Secrets
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Missing GOOGLE_API_KEY in Streamlit Secrets.")
    st.stop()

st.title("🎓 Monash University Course Advisor")
st.caption("Ask questions about undergraduate courses, entry pathways, and specialisations at Monash University Malaysia.")

# Cache the vector store and chain so it loads once in memory
@st.cache_resource
def load_rag_pipeline():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    
    vectorstore = Chroma(
        collection_name="monash_undergrad",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
    
    system_prompt = (
        "You are an academic advisor for prospective students of Monash University Malaysia.\n"
        "Answer questions strictly using the provided context below.\n"
        "If the answer is not in the context, state that you do not have that specific information "
        "and direct them to check monash.edu.my.\n"
        "Be concise, clear, and professional.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)

rag_chain = load_rag_pipeline()

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything about Monash Malaysia undergraduate programs."}
    ]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input field
if user_input := st.chat_input("e.g., What courses are offered under the School of Science?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching Monash course catalogs..."):
            result = rag_chain.invoke({"input": user_input})
            answer = result["answer"]
            st.markdown(answer)
            
            # Format source URLs into an expandable dropdown
            sources = sorted({d.metadata["source"] for d in result.get("context", []) if "source" in d.metadata})
            if sources:
                with st.expander("📚 Verified Sources"):
                    for src in sources:
                        st.markdown(f"- [{src}]({src})")
                        
    st.session_state.messages.append({"role": "assistant", "content": answer})
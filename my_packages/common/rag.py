import os
import sys
from langchain_ollama import OllamaEmbeddings
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(f"{script_dir}/../..")
results_dir = f"{project_dir}/experiments/few-shot/fox/testing_runs"

sys.path.append(project_dir)
print("Script is located in:", script_dir)
print("Project is located in:", project_dir)

class RagData:
    def __init__(
        self,
        embeddings: OllamaEmbeddings,
        language_docs: list[str],
        node_docs: list[str]
    ):
        self.embeddings: OllamaEmbeddings = embeddings
        self.language_docs: list[str] = language_docs
        self.node_docs: list[str] = node_docs

    def init_language_retriever(self):
        faiss_docs = [Document(page_content=doc["content"], metadata={"file": doc["file"], "chunk_id": doc["chunk_id"]}) for doc in self.language_docs]
        vectorstore = FAISS.from_documents(faiss_docs, self.embeddings)
        vectorstore.save_local("faiss_language_index") # Save FAISS index locally
        print("✅ FAISS index saved successfully.")
        self.language_retriever = vectorstore

    def init_node_retriever(self):
        faiss_docs = [Document(page_content=doc["content"], metadata={"file": doc["file"], "chunk_id": doc["chunk_id"]}) for doc in self.node_docs]
        vectorstore = FAISS.from_documents(faiss_docs, self.embeddings)
        vectorstore.save_local("faiss_node_index") # Save FAISS index locally
        print("✅ FAISS index saved successfully.")
        self.node_retriever = vectorstore
        
        
def init_rag_data(
    write_to_file: bool = False
)-> RagData:
    """Initializes a RagData object for the Midio documentation"""
    main_dataset_folder = f" {project_dir}/data/"
    df = pd.read_json(main_dataset_folder + 'rag_dataset.jsonl', lines=True)
    docs = df.to_dict(orient="records")
    # print(f"Number of docs: {len(docs)}")

    language_files = [
        # "overview.md",
        "the-midio-language.md", 
        "technical-details.midio", 
        "partial-function-application.md",
        "contexts.md",
        "loops.midio",
        "map-filter-reduce.md",
        "writing-tests.md",
    ]
    order_mapping = {name.upper(): idx for idx, name in enumerate(language_files)}

    language_docs = [doc for doc in docs if os.path.basename(doc["file"]).upper() in [f.upper() for f in language_files]]
    language_docs = sorted(
        language_docs,
        key=lambda doc: order_mapping.get(os.path.basename(doc["file"]).upper(), float('inf'))
    )

    node_files = [ # How to use
        "http.md",
        "std.md",
    ] 

    native_files = [  # Native implementation.
        "http_extern.midio",
        "std_extern.midio"
    ]
    # extern_func_info = Document("### External functions  \nExternal functions in Midio are functions with a native implementation in Rust. These functions generally provide better performance and access to lower-level system features compared to functions written directly in Midio. Most of the functions in the Midio standard library are external functions, which form a foundation of useful and efficient building blocks for your programs.  \n> You currently cannot create your own external functions, but this will be possible in the future.",)
    node_docs = [doc for doc in docs if os.path.basename(doc["file"]).upper() in [f.upper() for f in native_files]]
    print(f"Number of language docs: {len(language_docs)}")
    print(f"Number of node docs: {len(node_docs)}")

    rag_data = RagData(
        OllamaEmbeddings(model="nomic-embed-text"), 
        language_docs=language_docs,
        node_docs=node_docs
    )
    rag_data.init_language_retriever()
    rag_data.init_node_retriever()

    if write_to_file:
        formatted_language_context = "\n\n".join([doc["content"]for doc in language_docs])
        formatted_node_context = "\n\n".join([doc["content"]for doc in node_docs])
        # tokens = client(model=model["name"]).get_num_tokens(formatted_language_context) 
        # print(f"Number of tokens in the language context: {tokens}")

        with open("./language_doc", 'w') as f:
            f.write(formatted_language_context)

        with open("./node_doc", 'w') as f:
            f.write(formatted_node_context)
    return rag_data


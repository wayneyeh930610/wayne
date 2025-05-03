import os
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory

def qa_agent_with_local_faiss(question, chat_history, faiss_folder_path="faiss_index"):
    """
    使用本地端的 FAISS 向量資料庫進行查詢並產生回答。

    參數:
    - question: 使用者問題 (字串)
    - chat_history: 過往對話記錄 (可以是 list，也可以是 LangChain 的 memory 物件)
    - faiss_folder_path: 本地端 FAISS 向量資料庫的資料夾路徑
    """

    # 讀取環境變數中的 OpenAI API 金鑰
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("❌ 找不到 OpenAI API 金鑰，請確認已在環境變數中設定 OPENAI_API_KEY。")

    # 1. 初始化 GPT 模型
    model = ChatOpenAI(model="gpt-4", openai_api_key=openai_api_key)

    # 2. 讀取本地端的向量資料庫 (FAISS)
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=openai_api_key
    )

    try:
        db = FAISS.load_local(faiss_folder_path, embeddings_model, allow_dangerous_deserialization=True)
    except Exception as e:
        raise FileNotFoundError(f"❌ 找不到 FAISS 向量資料庫，請確認資料庫位置：{faiss_folder_path}\n錯誤訊息: {str(e)}")

    # 3. 建立檢索器 (retriever) #轉換為檢索器，設定相似度查詢的參數
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8, "score_threshold": 0.7} #k=8 表示回傳最相似的 8 筆資料
    )

    # 4. 建立 Conversational Retrieval Chain 記憶上下文對話歷史
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )#把檢索 + GPT 模型 + 記憶鏈接起來成一個問答系統。這就是RAG的流程

    # 5. 執行問答
    response = qa.invoke({
        "chat_history": chat_history,
        "question": question
    })

    return response


#pip install langchain langchain-community langchain-openai faiss-cpu openai


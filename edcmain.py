import streamlit as st
from qaz import create_vector_db_from_pdf  # 使用 qaz.py 中的函數
from wsx import qa_agent_with_local_faiss  # 已自動讀取環境變數的函數
import os  # 用來取得環境變數

def main():
    st.title("義守大學AI小幫手")

    # 取得環境變數中的 OpenAI API 金鑰
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("❌ 未偵測到 OpenAI API 金鑰，請設定環境變數 OPENAI_API_KEY。")
        return

    # 建立向量資料庫 (自動從固定 PDF 讀取)
    if st.button("建立向量資料庫"):
        result_msg = create_vector_db_from_pdf(faiss_folder_path="faiss_index")
        st.success(result_msg)

    # 進行問答查詢
    user_question = st.text_input("輸入你的問題：")
    if user_question and st.button("進行查詢"):
        response = qa_agent_with_local_faiss(
            question=user_question,
            chat_history=[],  # 傳入空的對話歷史
            faiss_folder_path="faiss_index"
        )
        if "answer" in response:
            st.write("回答：", response["answer"])
        else:
            st.write("❌ 沒有回答可以提供，請檢查資料庫是否有效。")

if __name__ == "__main__":
    main()

#pip install streamlit langchain langchain-community openai faiss-cpu

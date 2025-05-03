import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

def create_vector_db_from_pdf(faiss_folder_path="faiss_index"):
    """
    從本地的義守大學.pdf 建立向量資料庫，並儲存至指定資料夾。
    """

    # 取得 API 金鑰（從環境變數）
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("❌ 找不到環境變數 OPENAI_API_KEY，請先設定你的 OpenAI API 金鑰。")

    # 1. 直接指定 PDF 路徑
    temp_file_path = "../../OneDrive/桌面/義守大學.pdf"
    if not os.path.exists(temp_file_path):
        raise FileNotFoundError(f"❌ 找不到 PDF 檔案：{temp_file_path}")

    # 2. 解析 PDF #LangChain 提供的 PDF 載入器將 PDF 轉為結構化文本
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    # 3. 設定文本切分器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=550,
        chunk_overlap=100,
        separators=["\n", "。", "？", "!", ":", "，", "、", " "]
    )
    texts = text_splitter.split_documents(docs)

    # 4. 建立 Embeddings #將文本轉換成向量。
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=openai_api_key
    )

    # 5. 建立向量資料庫
    db = FAISS.from_documents(texts, embeddings_model)

    # 6. 儲存向量資料庫
    db.save_local(faiss_folder_path)

    print("✅ 成功建立並儲存向量資料庫！")
    print("➡️ 位置：", os.path.abspath(faiss_folder_path))
    print(f"📄 向量數量：{len(texts)}")

    return f"成功建立並儲存 {len(texts)} 筆向量至: {faiss_folder_path}"


# ✅ 主程式執行區
if __name__ == "__main__":
    result = create_vector_db_from_pdf()
    print(result)

#pip install langchain langchain-community faiss-cpu openai pymupdf tiktoken

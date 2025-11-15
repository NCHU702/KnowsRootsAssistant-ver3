"""
診斷 Layer 2 短文檔警告的工具

檢查：
1. 有多少 chunks 被跳過
2. 被跳過的內容是什麼
3. 是否影響檢索質量
"""

import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def diagnose_short_chunks(pdf_directory="./data", chunk_size=1000, chunk_overlap=200):
    """診斷短 chunk 問題"""
    
    print("="*80)
    print("Layer 2 短文檔警告診斷工具")
    print("="*80)
    print()
    
    # 初始化 splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # 統計
    total_pdfs = 0
    total_chunks = 0
    short_chunks = 0
    short_chunk_details = []
    
    # 遍歷所有 PDF
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
    
    print(f"📁 找到 {len(pdf_files)} 個 PDF 文件")
    print()
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        
        try:
            # 讀取 PDF
            reader = PdfReader(pdf_path)
            pdf_text = ""
            for page in reader.pages:
                pdf_text += page.extract_text() + "\n"
            
            # 分塊
            chunks = text_splitter.create_documents(
                texts=[pdf_text],
                metadatas=[{'paper_id': pdf_file}]
            )
            
            total_pdfs += 1
            
            # 檢查每個 chunk
            for i, chunk in enumerate(chunks):
                total_chunks += 1
                content_len = len(chunk.page_content)
                
                if content_len < 50:
                    short_chunks += 1
                    short_chunk_details.append({
                        'pdf': pdf_file,
                        'chunk_index': i,
                        'length': content_len,
                        'content': chunk.page_content[:100]  # 前 100 字元
                    })
        
        except Exception as e:
            print(f"⚠️  處理 {pdf_file} 時出錯: {e}")
            continue
    
    # 輸出結果
    print("="*80)
    print("📊 診斷結果")
    print("="*80)
    print()
    
    print(f"📚 處理的 PDF 數量: {total_pdfs}")
    print(f"📄 總 chunk 數量: {total_chunks}")
    print(f"⚠️  短 chunk 數量: {short_chunks} ({short_chunks/total_chunks*100:.2f}%)")
    print()
    
    # 判斷是否正常
    short_ratio = short_chunks / total_chunks if total_chunks > 0 else 0
    
    if short_ratio < 0.05:
        print("✅ 狀態: 正常")
        print("   短 chunk 比例很低，這是預期行為（封面、空白頁等）")
    elif short_ratio < 0.10:
        print("⚠️  狀態: 可接受")
        print("   短 chunk 比例稍高，但仍在合理範圍內")
    else:
        print("🔴 狀態: 需要關注")
        print("   短 chunk 比例過高，可能 PDF 提取有問題")
    
    print()
    print("="*80)
    print("📋 短 chunk 詳情（前 10 個）")
    print("="*80)
    print()
    
    for i, detail in enumerate(short_chunk_details[:10], 1):
        print(f"{i}. 文件: {detail['pdf']}")
        print(f"   Chunk 索引: {detail['chunk_index']}")
        print(f"   長度: {detail['length']} 字元")
        print(f"   內容預覽: {repr(detail['content'][:80])}")
        print()
    
    if len(short_chunk_details) > 10:
        print(f"... 還有 {len(short_chunk_details) - 10} 個短 chunk")
    
    print()
    print("="*80)
    print("💡 建議")
    print("="*80)
    print()
    
    if short_ratio < 0.05:
        print("✓ 無需採取行動")
        print("  短 chunk 主要來自封面、分隔頁等，跳過是正確的。")
    elif short_ratio < 0.10:
        print("• 可以保持現狀或考慮調整閾值")
        print("  當前閾值: 50 字元")
        print("  建議: 檢查被跳過的內容是否重要")
    else:
        print("⚠️  需要檢查 PDF 提取質量")
        print("  可能原因:")
        print("  1. PDF 是掃描版（需要 OCR）")
        print("  2. PDF 編碼問題")
        print("  3. PDF 包含大量圖表")
        print()
        print("  建議:")
        print("  - 檢查 PDF 文件的質量")
        print("  - 考慮使用更好的 PDF 提取工具（如 PyMuPDF）")
        print("  - 對掃描 PDF 使用 OCR")
    
    print()


if __name__ == "__main__":
    # 運行診斷
    diagnose_short_chunks()

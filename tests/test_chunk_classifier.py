"""
測試 Chunk Classifier 功能
"""

from system_api.chunk_classifier import ChunkClassifier, classify_chunk


# Sample chunk texts
SAMPLE_CHUNKS = {
    'summary': """
    Abstract: This paper presents a novel approach to time series forecasting
    using Long Short-Term Memory (LSTM) networks. We provide an overview of
    the methodology and introduce the main contributions of this work.
    """,
    
    'method': """
    3. Methodology
    
    We propose a hybrid architecture combining LSTM and mRBF networks.
    The LSTM model uses 3 layers with 128 hidden units each. The algorithm
    implements a sliding window approach with adaptive learning rate.
    """,
    
    'experiment': """
    4. Experimental Setup
    
    We evaluate our approach on three benchmark datasets: MNIST, CIFAR-10,
    and ImageNet. The training configuration uses batch size of 32, learning
    rate of 0.001, and Adam optimizer. We split the data into 80% training
    and 20% testing.
    """,
    
    'results': """
    5. Results and Evaluation
    
    Our model achieves 95.3% accuracy on the test set, outperforming the
    baseline by 3.2%. The precision is 0.94, recall is 0.96, and F1-score
    is 0.95. Performance metrics show consistent improvement across all
    evaluation criteria.
    """,
    
    'chinese_method': """
    本研究提出一種新的方法，結合 LSTM 和 mRBF 網路架構。
    我們的模型設計包含三個主要部分：特徵提取層、循環層和輸出層。
    演算法採用滑動視窗的方式處理時間序列數據。
    """,
    
    'chinese_experiment': """
    實驗設定如下：我們使用三個公開資料集進行評估。
    訓練參數包括批次大小 32、學習率 0.001。
    數據集被分為訓練集和測試集，比例為 8:2。
    """,
}


class TestChunkClassifier:
    """測試 ChunkClassifier 類別"""
    
    def setup_method(self):
        """每個測試方法前執行"""
        self.classifier = ChunkClassifier(llm=None, use_llm_fallback=False)
    
    def test_initialization(self):
        """測試初始化"""
        assert self.classifier is not None
        assert not self.classifier.use_llm_fallback
        assert len(self.classifier.compiled_patterns) == 4  # summary, method, experiment, results
    
    def test_classify_summary(self):
        """測試摘要分類"""
        result = self.classifier.classify_chunk(SAMPLE_CHUNKS['summary'])
        
        assert result['chunk_type'] == 'summary'
        assert result['confidence'] > 0.5
        assert result['method'] == 'heuristic'
        assert 'scores' in result
    
    def test_classify_method(self):
        """測試方法分類"""
        result = self.classifier.classify_chunk(SAMPLE_CHUNKS['method'])
        
        assert result['chunk_type'] == 'method'
        assert result['confidence'] > 0.5
        assert result['method'] == 'heuristic'
    
    def test_classify_experiment(self):
        """測試實驗分類"""
        result = self.classifier.classify_chunk(SAMPLE_CHUNKS['experiment'])
        
        assert result['chunk_type'] == 'experiment'
        assert result['confidence'] > 0.5
    
    def test_classify_results(self):
        """測試結果分類"""
        result = self.classifier.classify_chunk(SAMPLE_CHUNKS['results'])
        
        assert result['chunk_type'] == 'results'
        assert result['confidence'] > 0.5
    
    def test_classify_chinese_method(self):
        """測試中文方法分類"""
        result = self.classifier.classify_chunk(SAMPLE_CHUNKS['chinese_method'])
        
        assert result['chunk_type'] == 'method'
        assert result['confidence'] > 0.3  # Chinese might have lower confidence
    
    def test_classify_chinese_experiment(self):
        """測試中文實驗分類"""
        result = self.classifier.classify_chunk(SAMPLE_CHUNKS['chinese_experiment'])
        
        assert result['chunk_type'] == 'experiment'
        assert result['confidence'] > 0.3
    
    def test_classify_other(self):
        """測試其他類別（無關內容）"""
        text = "The quick brown fox jumps over the lazy dog. Lorem ipsum dolor sit amet."
        result = self.classifier.classify_chunk(text)
        
        assert result['chunk_type'] == 'other'
        assert result['confidence'] == 1.0  # Confident it's "other"
    
    def test_batch_classification(self):
        """測試批次分類"""
        from langchain_core.documents import Document
        
        chunks = [
            Document(page_content=SAMPLE_CHUNKS['summary'], metadata={'chunk_id': 'chunk_0'}),
            Document(page_content=SAMPLE_CHUNKS['method'], metadata={'chunk_id': 'chunk_1'}),
            Document(page_content=SAMPLE_CHUNKS['experiment'], metadata={'chunk_id': 'chunk_2'}),
        ]
        
        results = self.classifier.classify_chunks_batch(chunks)
        
        assert len(results) == 3
        assert results[0]['chunk_type'] == 'summary'
        assert results[1]['chunk_type'] == 'method'
        assert results[2]['chunk_type'] == 'experiment'
    
    def test_convenience_function(self):
        """測試便捷函數"""
        category = classify_chunk(SAMPLE_CHUNKS['method'])
        
        assert category == 'method'
        assert isinstance(category, str)
    
    def test_get_category_keywords(self):
        """測試獲取類別關鍵詞"""
        keywords = self.classifier.get_category_keywords('method')
        
        assert len(keywords) > 0
        assert any('method' in kw for kw in keywords)


def test_all_categories():
    """整合測試：所有類別"""
    classifier = ChunkClassifier()
    
    expected_categories = ['summary', 'method', 'experiment', 'results']
    
    for category in expected_categories:
        if category in SAMPLE_CHUNKS:
            result = classifier.classify_chunk(SAMPLE_CHUNKS[category])
            print(f"✓ {category}: {result['chunk_type']} (confidence: {result['confidence']:.2f})")
            assert result['chunk_type'] == category


if __name__ == "__main__":
    print("🧪 Running Chunk Classifier Tests")
    print("=" * 70)
    
    # Run basic functionality test
    test_all_categories()
    
    print("\n" + "=" * 70)
    print("✅ All manual tests passed!")
    print("\nRun 'pytest tests/test_chunk_classifier.py -v' for full test suite")

import pytest
from backend.agent.graph import create_agent_workflow
from backend.agent.state import create_initial_state, InputType
from backend.tasks.summarize import summarize_text
from backend.tasks.sentiment import analyze_sentiment
from backend.tasks.code_explain import explain_code


class TestContentExtraction:
    """Test content extraction from various sources"""
    
    def test_text_input(self):
        """Test basic text input"""
        workflow = create_agent_workflow()
        
        state = create_initial_state(
            input_type=InputType.TEXT,
            raw_input="Please summarize this: AI is transforming the world."
        )
        
        result = workflow.invoke(state)
        
        assert result['extracted_text'] != ""
        assert 'errors' not in result or len(result['errors']) == 0
    
    def test_youtube_url_detection(self):
        """Test YouTube URL detection and transcript fetching"""
        from backend.extractors.youtube import detect_youtube_url, extract_youtube_video_id
        
        text = "Check this out: https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        
        assert detect_youtube_url(text) == True
        
        video_id = extract_youtube_video_id(text)
        assert video_id == "dQw4w9WgXcQ"


class TestIntentClassification:
    """Test intent classification"""
    
    def test_explicit_summarize_request(self):
        """Test when user explicitly asks for summary"""
        workflow = create_agent_workflow()
        
        state = create_initial_state(
            input_type=InputType.TEXT,
            raw_input="Summarize this article: AI is revolutionizing healthcare..."
        )
        
        result = workflow.invoke(state)
        
        assert result['detected_task'] == 'summarize'
        assert result['confidence'] > 0.7
    
    def test_explicit_sentiment_request(self):
        """Test when user explicitly asks for sentiment"""
        workflow = create_agent_workflow()
        
        state = create_initial_state(
            input_type=InputType.TEXT,
            raw_input="What's the sentiment of this: I love this product!"
        )
        
        result = workflow.invoke(state)
        
        assert result['detected_task'] in ['sentiment', 'qa']
        
    def test_unclear_intent_triggers_followup(self):
        """Test that unclear intent triggers follow-up question"""
        workflow = create_agent_workflow()
        
        # Just extracting content without clear instruction
        state = create_initial_state(
            input_type=InputType.TEXT,
            raw_input="Here is some text about machine learning."
        )
        
        result = workflow.invoke(state)
        
        # Should trigger follow-up due to low confidence
        assert result.get('needs_clarification') == True or result['confidence'] < 0.7


class TestSummarization:
    """Test summarization task"""
    
    def test_summary_format(self):
        """Test that summary has all three required formats"""
        text = """
        Artificial Intelligence has made significant progress in recent years.
        Machine learning algorithms can now process vast amounts of data.
        Deep learning has enabled breakthroughs in computer vision and NLP.
        AI is being applied in healthcare, finance, and many other industries.
        The future of AI holds both exciting opportunities and challenges.
        """
        
        result = summarize_text(text)
        
        assert 'one_liner' in result
        assert 'bullets' in result
        assert 'five_sentences' in result
        
        assert len(result['bullets']) == 3
        assert isinstance(result['one_liner'], str)
        assert len(result['one_liner']) > 0


class TestSentimentAnalysis:
    """Test sentiment analysis task"""
    
    def test_positive_sentiment(self):
        """Test detection of positive sentiment"""
        text = "This is absolutely amazing! I love it so much!"
        
        result = analyze_sentiment(text)
        
        assert result['label'] == 'positive'
        assert result['confidence'] > 0.5
        assert 'justification' in result
    
    def test_negative_sentiment(self):
        """Test detection of negative sentiment"""
        text = "This is terrible. I hate it and it's completely broken."
        
        result = analyze_sentiment(text)
        
        assert result['label'] == 'negative'
        assert result['confidence'] > 0.5
    
    def test_neutral_sentiment(self):
        """Test detection of neutral sentiment"""
        text = "The meeting is scheduled for 3 PM on Tuesday."
        
        result = analyze_sentiment(text)
        
        assert result['label'] in ['neutral', 'positive', 'negative']
        assert 'confidence' in result


class TestCodeExplanation:
    """Test code explanation task"""
    
    def test_python_code_explanation(self):
        """Test explanation of Python code"""
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
        """
        
        result = explain_code(code, "python")
        
        assert result['success'] == True
        assert 'explanation' in result
        assert 'bugs' in result
        assert 'time_complexity' in result
        assert len(result['explanation']) > 0
    
    def test_code_language_detection(self):
        """Test language detection in code"""
        from backend.extractors.ocr import detect_code_in_text
        
        python_code = "def hello():\n    print('Hello')"
        result = detect_code_in_text(python_code)
        
        assert result['is_code'] == True
        assert result['language'] == 'python'


class TestErrorHandling:
    """Test error handling and robustness"""
    
    def test_empty_input(self):
        """Test handling of empty input"""
        workflow = create_agent_workflow()
        
        state = create_initial_state(
            input_type=InputType.TEXT,
            raw_input=""
        )
        
        result = workflow.invoke(state)
        
        # Should either ask for clarification or handle gracefully
        assert 'errors' in result or result.get('needs_clarification')
    
    def test_invalid_file_path(self):
        """Test handling of invalid file path"""
        workflow = create_agent_workflow()
        
        state = create_initial_state(
            input_type=InputType.IMAGE,
            raw_input=None,
            file_path="/nonexistent/file.jpg"
        )
        
        result = workflow.invoke(state)
        
        # Should capture error
        assert len(result.get('errors', [])) > 0


class TestEndToEnd:
    """End-to-end integration tests"""
    
    def test_text_summarization_workflow(self):
        """Test complete workflow for text summarization"""
        workflow = create_agent_workflow()
        
        state = create_initial_state(
            input_type=InputType.TEXT,
            raw_input="""Summarize: Climate change is one of the most pressing 
            issues of our time. Global temperatures are rising due to greenhouse 
            gas emissions. Scientists warn that we must act quickly to reduce 
            carbon emissions and transition to renewable energy sources."""
        )
        
        result = workflow.invoke(state)
        
        assert result['current_step'] in ['task_completed', 'awaiting_clarification']
        
        if result['current_step'] == 'task_completed':
            assert 'result' in result
            assert result['result']['success'] == True


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
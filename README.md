# 🎯 Smart Resume Matcher

*An AI-powered hiring system that helps managers easily find and evaluate the right candidates*

## 🚀 What This Does

Transform your hiring process with an AI assistant that understands both job requirements and candidate profiles at a human level. Instead of manually sifting through hundreds of resumes or relying on basic keyword matching, get intelligent recommendations with detailed analysis in seconds.

**Core Capabilities:**
- **Natural Language Job Matching**: Describe what you're looking for in plain English
- **Intelligent Candidate Ranking**: AI-powered scoring based on job fit and qualifications  
- **Contextual Resume Analysis**: Deep understanding of skills, experience, and career progression
- **Interactive Candidate Exploration**: Ask follow-up questions and compare candidates dynamically


## 🏗️ Architecture Overview

### Dual-Mode Intelligence System

**🔍 Similarity Search Mode**
- Processes job descriptions through advanced NLP
- Generates multiple search perspectives using RAG Fusion
- Ranks candidates using reciprocal rank fusion
- Returns contextually relevant matches

**🎯 Direct Lookup Mode**  
- Instant candidate profile retrieval by ID
- Detailed resume analysis and insights
- Cross-candidate comparison capabilities

### Technical Foundation

```
User Query → Query Classification → Retrieval Strategy → LLM Analysis → Intelligent Response
     ↓              ↓                    ↓               ↓              ↓
Natural Language   Job Description    Vector Search    GPT Analysis   Ranked Results
   or ID List      vs ID Request      + RAG Fusion    + Context      + Explanations
```

**Technology Stack:**
- **AI/ML**: OpenAI GPT-4, HuggingFace Transformers, Sentence Transformers
- **Vector Database**: FAISS with cosine similarity
- **Framework**: LangChain for orchestration
- **Interface**: Streamlit for interactive web UI
- **Evaluation**: RAGAS metrics for quality assessment

## 🛠️ Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your OpenAI API key to .env file
```

### Running Locally

```bash
# Launch the web interface
streamlit run demo/interface.py

# Or run individual components
python demo/ingest_data.py    # Process resume data
python demo/retriever.py      # Test retrieval system
```


## 🔬 Research & Evaluation

This system has been rigorously evaluated using multiple metrics:

- **Context Precision**: How relevant are retrieved resumes?
- **Context Recall**: Are all relevant candidates found?
- **Faithfulness**: Do AI responses accurately reflect resume content?
- **Answer Similarity**: Consistency across similar queries

**Evaluation Framework**: RAGAS (Retrieval Augmented Generation Assessment)

## 🎨 Usage Examples

### Job Description Matching
```python
query = """
Looking for a senior full-stack developer with:
- 5+ years React and Node.js experience  
- Experience with cloud platforms (AWS/Azure)
- Team leadership background
- Startup environment experience preferred
"""
# Returns ranked candidates with detailed fit analysis
```

### Candidate Comparison
```python
query = "Compare candidates 101, 205, and 312 for the product manager role"
# Returns side-by-side analysis with strengths/weaknesses
```

### Skills-Based Search
```python
query = "Find data scientists with deep learning and healthcare domain experience"
# Returns candidates ranked by technical skills and industry background
```

## 🤝 Contributing

We welcome contributions! This project serves as a foundation for exploring AI in recruitment.

**Areas for Enhancement:**
- Additional retrieval strategies
- Multi-language resume support  
- Integration with ATS systems
- Advanced bias detection
- Real-time candidate scoring

**Getting Started:**
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## 📄 License & Citation

This project is part of academic research in AI-powered recruitment systems.

## 🙏 Acknowledgments

- **RAG Fusion**: Advanced retrieval methodology
- **LangChain Community**: Framework and tools
- **Streamlit**: Rapid prototyping platform
- **OpenAI**: Language model capabilities

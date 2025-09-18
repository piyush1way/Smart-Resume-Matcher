import sys
sys.dont_write_bytecode = True

import streamlit as st
import numpy as np

def render(document_list: list, meta_data: dict, time_elapsed: float):
    """Show processing details in a clean, simple format"""
    
    message_map = {
        "retrieve_applicant_jd": "🎯 **Job Description Query** - Found matching candidates using AI similarity search",
        "retrieve_applicant_id": "🔍 **Applicant ID Query** - Retrieved specific candidate information", 
        "no_retrieve": "💬 **General Question** - Answered using conversation context"
    }
    
    with st.expander(f"📊 **Processing Details** ({np.round(time_elapsed, 2)}s)", expanded=False):
        
        # Query type and processing info
        query_type = meta_data.get('query_type', 'unknown')
        st.info(message_map.get(query_type, "Processing completed"))
        
        # Show basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⏱️ Time", f"{np.round(time_elapsed, 2)}s")
        with col2:
            st.metric("🧠 Mode", meta_data.get('rag_mode', 'Standard'))
        with col3:
            st.metric("📋 Results", len(document_list) if document_list else 0)
        
        # Show retrieved candidates if available
        if document_list and len(document_list) > 0:
            st.markdown("### 👥 Retrieved Candidates")
            
            # Create tabs for better organization
            tab1, tab2 = st.tabs(["📄 Candidate Previews", "🔧 Technical Details"])
            
            with tab1:
                # Show candidate previews
                for i, document in enumerate(document_list[:5], 1):
                    # Extract candidate ID
                    candidate_id = "Unknown"
                    if "Applicant ID" in document:
                        try:
                            candidate_id = document.split("Applicant ID")[1].split("\n")[0].strip()
                        except:
                            candidate_id = f"Candidate {i}"
                    
                    with st.expander(f"👤 {candidate_id}", expanded=False):
                        st.text_area(
                            "Resume Content", 
                            document, 
                            height=200, 
                            key=f"resume_{i}",
                            disabled=True
                        )
            
            with tab2:
                # Technical processing details
                if query_type == "retrieve_applicant_jd":
                    if 'extracted_input' in meta_data:
                        st.markdown("**🎯 Extracted Query:**")
                        st.code(meta_data['extracted_input'])
                    
                    if 'subquestion_list' in meta_data and meta_data['subquestion_list']:
                        st.markdown("**❓ Generated Sub-queries:**")
                        for i, subq in enumerate(meta_data['subquestion_list'], 1):
                            st.markdown(f"{i}. {subq}")
                    
                    if 'retrieved_docs_with_scores' in meta_data:
                        st.markdown("**📊 Similarity Scores:**")
                        scores = meta_data['retrieved_docs_with_scores']
                        if isinstance(scores, dict) and scores:
                            # Show top scores in a simple format
                            for i, (doc_id, score) in enumerate(list(scores.items())[:5], 1):
                                st.markdown(f"{i}. **ID {doc_id}**: {score:.4f}")
                
                elif query_type == "retrieve_applicant_id":
                    st.markdown("**🔍 Requested Candidate IDs:**")
                    st.code(str(meta_data.get('extracted_input', 'N/A')))
        
        else:
            st.warning("No candidates retrieved for this query")

if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2])
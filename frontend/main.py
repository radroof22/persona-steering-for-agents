import streamlit as st
import pandas as pd
import asyncio
import logging
import threading
import time
from typing import Dict, Any

from config import STREAMLIT_CONFIG, COLORS
from api_client import APIClient
from utils import (
    validate_csv_format, 
    validate_users_format,
    validate_queries_format,
    process_csv_data, 
    process_separate_csv_data,
    create_results_dataframe,
    create_new_results_dataframe,
    format_cluster_display,
    check_api_connection
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(**STREAMLIT_CONFIG)

# Custom CSS for styling
st.markdown(f"""
<style>
    .main {{
        background-color: {COLORS['background']};
    }}
    .stButton > button {{
        background-color: {COLORS['primary']};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }}
    .stButton > button:hover {{
        background-color: {COLORS['secondary']};
    }}
    .cluster-header {{
        color: {COLORS['primary']};
        font-weight: 600;
        margin-bottom: 1rem;
    }}
    .query-container {{
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid {COLORS['secondary']};
        margin-bottom: 0.5rem;
        color: {COLORS['text']};
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .query-container strong {{
        color: {COLORS['primary']};
    }}
    .error-container {{
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff4444;
        margin-bottom: 0.5rem;
        color: #333333;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .success-indicator {{
        color: #00BFA5;
        font-weight: bold;
    }}
    .error-indicator {{
        color: #ff4444;
        font-weight: bold;
    }}
    .unique-query-container {{
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin-bottom: 0.5rem;
        color: {COLORS['text']};
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .unique-query-container strong {{
        color: #856404;
    }}
    .unique-indicator {{
        color: #856404;
        font-weight: bold;
    }}
</style>
""", unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'show_developer_output' not in st.session_state:
        st.session_state.show_developer_output = False
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    # Header
    st.title("🧠 Clustered Prompt Personalization Demo")
    st.markdown("**Visualize how we group and rewrite prompts based on your persona.**")
    
    # Check API connection
    api_client = APIClient()
    api_connected = check_api_connection(api_client)
    
    if not api_connected:
        st.error("⚠️ Cannot connect to the API. Please ensure the backend server is running on http://localhost:8000")
        st.stop()
    
    # Main content
    if st.session_state.results is None:
        show_landing_page(api_client)
    else:
        show_results_page(api_client)


def show_landing_page(api_client: APIClient):
    """Display the landing page with input forms."""
    
    st.markdown("---")
    
    # Initialize session state for DataFrames
    if 'users_df' not in st.session_state:
        st.session_state.users_df = None
    if 'queries_df' not in st.session_state:
        st.session_state.queries_df = None
    if 'current_users_file' not in st.session_state:
        st.session_state.current_users_file = None
    if 'current_queries_file' not in st.session_state:
        st.session_state.current_queries_file = None
    
    # File uploads
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Upload Users")
        users_file = st.file_uploader(
            "Upload users CSV file",
            type=['csv'],
            help="CSV with columns: user_id, description"
        )
        
        if users_file:
            # Check if file has changed
            if st.session_state.current_users_file != users_file.name:
                st.session_state.users_df = None
                st.session_state.current_users_file = users_file.name
            
            try:
                # Only read if not already in session state
                if st.session_state.users_df is None:
                    st.session_state.users_df = pd.read_csv(users_file)
                
                users_df = st.session_state.users_df
                st.success(f"✅ Loaded {len(users_df)} users")
                if st.checkbox("Preview users"):
                    st.dataframe(users_df.head())
            except Exception as e:
                st.error(f"❌ Error reading users file: {str(e)}")
                st.session_state.users_df = None
    
    with col2:
        st.subheader("📝 Upload Queries")
        queries_file = st.file_uploader(
            "Upload queries CSV file",
            type=['csv'],
            help="CSV with columns: user_id, question"
        )
        
        if queries_file:
            # Check if file has changed
            if st.session_state.current_queries_file != queries_file.name:
                st.session_state.queries_df = None
                st.session_state.current_queries_file = queries_file.name
            
            try:
                # Only read if not already in session state
                if st.session_state.queries_df is None:
                    st.session_state.queries_df = pd.read_csv(queries_file)
                
                queries_df = st.session_state.queries_df
                st.success(f"✅ Loaded {len(queries_df)} queries")
                if st.checkbox("Preview queries"):
                    st.dataframe(queries_df.head())
            except Exception as e:
                st.error(f"❌ Error reading queries file: {str(e)}")
                st.session_state.queries_df = None
    
    # Strategy selection (disabled for now as per PRD)
    st.subheader("Rewrite Strategy")
    strategy = st.selectbox(
        "Choose how to personalize your responses:",
        ["LLM - Reprompted"],
        disabled=True,
        help="Additional strategies coming soon"
    )
    
    
    
    # Submit button
    if st.button("🚀 Process Prompts", type="primary", disabled=st.session_state.processing):
        if st.session_state.users_df is None:
            st.error("Please upload a users CSV file.")
            return
        
        if st.session_state.queries_df is None:
            st.error("Please upload a queries CSV file.")
            return
        
        # Get DataFrames from session state
        users_df = st.session_state.users_df
        queries_df = st.session_state.queries_df
        
        # Validate files
        users_valid, users_error = validate_users_format(users_df)
        queries_valid, queries_error = validate_queries_format(queries_df)
        
        if not users_valid:
            st.error(f"❌ Users file error: {users_error}")
            return
        
        if not queries_valid:
            st.error(f"❌ Queries file error: {queries_error}")
            return
        
        # Set processing state
        st.session_state.processing = True
        st.rerun()
    
    # Processing section
    if st.session_state.processing:
        st.markdown("---")
        st.subheader("🔄 Processing Your Prompts")
        st.info("⏱️ **Note:** Processing may take 1-2 minutes on CPU. The system will cluster your queries and generate personalized rewrites for each user.")
        
        # Process data for API
        users_df = st.session_state.users_df
        queries_df = st.session_state.queries_df
        users, queries = process_separate_csv_data(users_df, queries_df)
        
        # Show processing status
        with st.spinner("🔄 Processing your prompts (this may take 1-2 minutes on CPU)..."):
            # Call API
            try:
                # Direct synchronous call - no event loop needed
                results = api_client.generate_personalized_rewrites(users, queries)
                
                # Store results in session state
                st.session_state.results = results
                st.session_state.users_df = users_df
                st.session_state.queries_df = queries_df
                st.session_state.processing = False
                
                # Redirect to results page
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error processing prompts: {str(e)}")
                logger.error(f"API call failed: {e}")
                st.session_state.processing = False
                st.rerun()





def show_results_page(api_client: APIClient):
    """Display the results page with clustered and rewritten prompts."""
    
    results = st.session_state.results
    users_df = st.session_state.users_df
    queries_df = st.session_state.queries_df
    
    # Back button
    if st.button("← Back to Upload"):
        st.session_state.results = None
        st.rerun()
    
    st.markdown("---")
    
    # Results header
    st.subheader("📊 Your Personalized Results")
    st.markdown(f"**Total Users:** {len(users_df)}")
    st.markdown(f"**Total Queries:** {len(queries_df)}")
    
    # Developer output toggle
    st.session_state.show_developer_output = st.checkbox(
        "Show developer output",
        value=st.session_state.show_developer_output
    )
    
    # Display results - new API format
    personalized_results = results.get('results', [])
    
    if personalized_results:
        st.subheader("🎯 Personalized Query Results")
        
        # Group results by cluster for better organization
        clusters_dict = {}
        for result in personalized_results:
            cluster_id = result.get('cluster_id', -1)
            if cluster_id not in clusters_dict:
                clusters_dict[cluster_id] = []
            clusters_dict[cluster_id].append(result)
        
        # Separate unique queries from regular clusters
        unique_queries = []
        regular_clusters = {}
        
        for cluster_id, cluster_results in clusters_dict.items():
            if len(cluster_results) == 1:
                # Collect all unique queries
                unique_queries.extend(cluster_results)
            else:
                # Keep regular clusters as they are
                regular_clusters[cluster_id] = cluster_results
        
        # Display unique queries in one expander
        if unique_queries:
            st.warning("⚠️ **Note:** These queries couldn't be grouped with others. They're processed individually, which is less efficient than grouped processing.")
            
            with st.expander(f"⚠️ Unique Queries ({len(unique_queries)} queries)", expanded=True):
                for result in unique_queries:
                    original_query = result.get('original_query', 'N/A')
                    summarized_query = result.get('summarized_query', 'N/A')
                    summarized_response = result.get('summarized_response', 'N/A')
                    personalized_response = result.get('personalized_response', 'N/A')
                    success = result.get('success', False)
                    error_message = result.get('error_message', '')
                    user_id = result.get('user_id', 'Unknown')
                    
                    # Display query result
                    if success:
                        st.markdown(f"""
                        <div class="unique-query-container">
                            <span class="unique-indicator">⚠️ UNIQUE QUERY</span><br>
                            <strong>User:</strong> {user_id}<br>
                            <strong>Original Query:</strong> {original_query}<br>
                            <strong>Cluster Summary:</strong> {summarized_query}<br>
                            <strong>General Response:</strong> {summarized_response}<br>
                            <strong>Personalized Response:</strong> {personalized_response}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="error-container">
                            <span class="error-indicator">❌ FAILED</span><br>
                            <strong>User:</strong> {user_id}<br>
                            <strong>Original Query:</strong> {original_query}<br>
                            <strong>Error:</strong> {error_message}
                        </div>
                        """, unsafe_allow_html=True)
        
        # Display regular clusters
        for cluster_id, cluster_results in regular_clusters.items():
            cluster_title = f"Cluster {cluster_id} ({len(cluster_results)} queries)"
            
            with st.expander(cluster_title, expanded=True):
                for result in cluster_results:
                    original_query = result.get('original_query', 'N/A')
                    summarized_query = result.get('summarized_query', 'N/A')
                    summarized_response = result.get('summarized_response', 'N/A')
                    personalized_response = result.get('personalized_response', 'N/A')
                    success = result.get('success', False)
                    error_message = result.get('error_message', '')
                    user_id = result.get('user_id', 'Unknown')
                    
                    # Display query result
                    if success:
                        st.markdown(f"""
                        <div class="query-container">
                            <span class="success-indicator">✅ SUCCESS</span><br>
                            <strong>User:</strong> {user_id}<br>
                            <strong>Original Query:</strong> {original_query}<br>
                            <strong>Cluster Summary:</strong> {summarized_query}<br>
                            <strong>General Response:</strong> {summarized_response}<br>
                            <strong>Personalized Response:</strong> {personalized_response}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="error-container">
                            <span class="error-indicator">❌ FAILED</span><br>
                            <strong>User:</strong> {user_id}<br>
                            <strong>Original Query:</strong> {original_query}<br>
                            <strong>Error:</strong> {error_message}
                        </div>
                        """, unsafe_allow_html=True)
    
    # Download section
    st.markdown("---")
    st.subheader("💾 Download Results")
    
    # Create results DataFrame from new API format
    if personalized_results:
        results_df = create_new_results_dataframe(personalized_results)
        
        # Download button
        csv_data = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download results as CSV",
            data=csv_data,
            file_name="personalized_prompts_results.csv",
            mime="text/csv"
        )
    else:
        st.info("No results to download yet.")
    
    # Developer output section
    if st.session_state.show_developer_output:
        st.markdown("---")
        st.subheader("🔧 Developer Output")
        
        # Raw API response
        with st.expander("Raw API Response"):
            st.json(results)
        
        # Clusters endpoint response
        with st.expander("Clusters API Response"):
            try:
                users, queries = process_separate_csv_data(users_df, queries_df)
                clusters_response = api_client.get_clusters(users, queries)
                st.json(clusters_response)
            except Exception as e:
                st.error(f"Error fetching clusters data: {e}")


if __name__ == "__main__":
    main() 
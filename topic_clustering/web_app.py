import streamlit as st
import streamlit.components.v1 as components
import json
import numpy as np

###############################################################################
#                          SET YOUR TAXONOMY FILE HERE
###############################################################################
TAXONOMY_FILE_PATH = "/Users/jashshah/Desktop/science-cartography-1/topic_clustering/taxonomies/results/mapped_taxonomy.json"
###############################################################################
#                         HELPER FUNCTIONS
###############################################################################
def load_taxonomy_data(filepath):
    """Load the nested JSON taxonomy from the given file path."""
    with open(filepath, 'r') as f:
        return json.load(f)

def is_paper_level(node):
    """
    A node is considered 'paper level' if it has 'Papers' and
    no other subtopic keys.
    """
    if not isinstance(node, dict):
        return False
    keys_excluding_papers = [k for k in node.keys() if k != "Papers"]
    return "Papers" in node and len(keys_excluding_papers) == 0

def get_subtopics_and_papers(node):
    """
    Returns (list_of_subtopics, list_of_papers).
    - subtopics: list of (subtopic_key, subnode_dict)
    - papers: list of paper objects (with Title, Abstract, Rationale).
    """
    if not isinstance(node, dict):
        return [], []
    
    subtopics = []
    papers = []
    
    if "Papers" in node and isinstance(node["Papers"], list):
        papers = node["Papers"]
    
    for key, val in node.items():
        if key == "Papers":
            continue
        subtopics.append((key, val))
    
    return subtopics, papers

def navigate_path(taxonomy, path):
    """
    Traverse the taxonomy by following the keys in 'path'.
    Returns the node (dict) at the end of that path.
    """
    current = taxonomy
    for key in path:
        if not isinstance(current, dict):
            return {}
        current = current.get(key, {})
    return current

def count_papers_recursive(node):
    """
    Recursively count how many total papers are under this node.
    """
    if not isinstance(node, dict):
        return 0
    
    total = 0
    if "Papers" in node and isinstance(node["Papers"], list):
        total += len(node["Papers"])
    
    for k, v in node.items():
        if k == "Papers":
            continue
        if isinstance(v, dict):
            total += count_papers_recursive(v)
    return total

def list_subtopic_paper_counts(node):
    """
    For immediate subtopics of 'node', return a list of their paper counts (recursive).
    Used to compute stats (mean, median, etc.).
    """
    subtopics, _ = get_subtopics_and_papers(node)
    counts = [count_papers_recursive(subnode) for _, subnode in subtopics]
    return counts

def get_cluster_statistics(node):
    """
    Show cluster-level stats for immediate subtopics:
      - Total Clusters
      - Total Papers
      - Average, median, std, etc.
    """
    subtopics, _papers = get_subtopics_and_papers(node)
    cluster_count = len(subtopics)
    paper_counts = list_subtopic_paper_counts(node)
    
    if paper_counts:
        stats = {
            'Total Clusters': cluster_count,
            'Total Papers': sum(paper_counts),
            'Average Papers per Cluster': round(np.mean(paper_counts), 2),
            'Median Papers': round(np.median(paper_counts), 2),
            'Standard Deviation': round(np.std(paper_counts), 2),
            'Max Papers in Cluster': max(paper_counts),
            'Min Papers in Cluster': min(paper_counts)
        }
    else:
        stats = {
            'Total Clusters': cluster_count,
            'Total Papers': 0,
            'Average Papers per Cluster': 0,
            'Median Papers': 0,
            'Standard Deviation': 0,
            'Max Papers in Cluster': 0,
            'Min Papers in Cluster': 0
        }
    return stats

def compute_depth(node):
    """
    Recursively compute the maximum depth of the taxonomy branch.
    If node is paper-level, depth is 0.
    """
    if not isinstance(node, dict) or is_paper_level(node):
        return 0
    max_child_depth = 0
    for key, subnode in node.items():
        if key == "Papers":
            continue
        if isinstance(subnode, dict):
            max_child_depth = max(max_child_depth, compute_depth(subnode))
    return 1 + max_child_depth

def average_abstract_length(papers):
    """
    Computes the average abstract length (in words) for a list of papers.
    """
    lengths = []
    for paper in papers:
        abstract = paper.get("Abstract", "")
        word_count = len(abstract.split())
        lengths.append(word_count)
    if lengths:
        return round(np.mean(lengths), 2)
    return 0

###############################################################################
#                        PATH DETAIL CARDS (NON-LEAF)
###############################################################################
def display_path_details(path, taxonomy):
    """
    Displays fancy cards for each item in 'path' (arbitrary depth)
    using Streamlit components to render HTML.
    For each intermediate node, we also show:
      - number of subtopics
      - total papers (recursive)
      - branch depth from that node
    """
    if not path:
        return
    
    st.markdown("### Path Details 🔎")
    current_node = taxonomy
    
    for i, key in enumerate(path):
        level = i + 1
        indent = 32 * i  # Increase indent with depth
        
        node_data = current_node.get(key, {})
        subtopics, _ = get_subtopics_and_papers(node_data)
        subtopic_count = len(subtopics)
        paper_count = count_papers_recursive(node_data)
        branch_depth = compute_depth(node_data)
        
        html_code = f"""
            <div style='
                background: linear-gradient(135deg, #1F2937 0%, #2D3748 100%);
                border-radius: 8px; 
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 
                            0 2px 4px -1px rgba(0, 0, 0, 0.06);
                border: 1px solid #475569;
                padding: 24px; 
                margin-left: {indent}px;
                margin-bottom: 16px;
                transition: all 0.2s ease-in-out;
            '
            onmouseover="this.style.transform='scale(1.01)';"
            onmouseout="this.style.transform='scale(1.0)';">
                
                <div style='
                    display: flex; 
                    align-items: center; 
                    justify-content: space-between;
                    margin-bottom: 16px; 
                    padding-bottom: 12px; 
                    border-bottom: 1px solid #4B5563;
                '>
                    <div style='display: flex; align-items: center;'>
                        <div style='font-size: 1.2rem; margin-right: 8px;'>📂</div>
                        <h4 style='
                            font-size: 1.25rem; 
                            font-weight: 600; 
                            color: #E2E8F0; 
                            margin: 0;
                        '>
                            {key}
                        </h4>
                    </div>
                    <span style='
                        background-color: #F59E0B; 
                        color: #1F2937; 
                        padding: 8px 16px; 
                        border-radius: 9999px; 
                        font-weight: 700; 
                        font-size: 1.125rem;
                    '>
                        🚀 Level {level}
                    </span>
                </div>
                
                <div style='color: #CBD5E1; font-size: 0.95rem;'>
                    <p style='margin: 0;'>
                        <strong>📁 Subtopics:</strong> {subtopic_count}
                    </p>
                    <p style='margin: 4px 0 0 0;'>
                        <strong>📰 Total Papers (recursive):</strong> {paper_count}
                    </p>
                    <p style='margin: 4px 0 0 0;'>
                        <strong>🌳 Branch Depth:</strong> {branch_depth}
                    </p>
                </div>
            </div>
        """
        components.html(html_code, height=200)
        
        if isinstance(node_data, dict):
            current_node = node_data

###############################################################################
#                             STREAMLIT MAIN
###############################################################################
def main():
    st.set_page_config(layout="wide")
    st.title('Science Cartography - Taxonomy Exploration ✨')
    
    try:
        data = load_taxonomy_data(TAXONOMY_FILE_PATH)
        
        if 'path' not in st.session_state:
            st.session_state.path = []
        
        if st.button('← Back') and st.session_state.path:
            st.session_state.path.pop()
            st.rerun()
        
        current_node = navigate_path(data, st.session_state.path)
        leaf_is_papers = is_paper_level(current_node)
        
        # For non-leaf nodes, show a summary dashboard.
        if not leaf_is_papers:
            st.subheader('Topic Metrics')
            # Total papers assigned (recursively)
            total_papers = count_papers_recursive(current_node)
            # Immediate subtopic count
            subtopics, _ = get_subtopics_and_papers(current_node)
            subtopic_count = len(subtopics)
            # Compute branch depth from current node
            branch_depth = compute_depth(current_node)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Papers Assigned", total_papers)
            col2.metric("Subtopics", subtopic_count)
            col3.metric("Branch Depth", branch_depth)
        
        if st.session_state.path:
            path_str = " > ".join(st.session_state.path)
            st.markdown(f"**Current Path**: {path_str}")
        
        display_path_details(st.session_state.path, data)
        st.markdown("<div style='margin-bottom: 48px;'></div>", unsafe_allow_html=True)
        
        # For leaf nodes, show additional metrics
        if leaf_is_papers:
            st.subheader("Papers Metrics 📑")
            papers = current_node.get("Papers", [])
            num_papers = len(papers)
            avg_abstract = average_abstract_length(papers)
            col1, col2 = st.columns(2)
            col1.metric("Number of Papers", num_papers)
            col2.metric("Average Abstract Length", f"{avg_abstract} words")
            
            # Display papers in a tabbed interface.
            if papers:
                paper_tabs = st.tabs([f"Paper {i+1}" for i in range(len(papers))])
                for idx, paper in enumerate(papers):
                    with paper_tabs[idx]:
                        title = paper.get("Title", "Untitled Paper")
                        abstract = paper.get("Abstract", "No Abstract provided.")
                        rationale = paper.get("Rationale", "")
                        
                        paper_html = f"""
                        <div style="
                            background: linear-gradient(135deg, #1F2937 0%, #2D3748 100%);
                            border-radius: 12px;
                            border: 1px solid #475569;
                            margin-bottom: 16px;
                            padding: 16px;
                            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1),
                                        0 2px 4px -1px rgba(0, 0, 0, 0.06);
                            transition: all 0.2s ease-in-out;
                        ">
                            <h3 style="
                                color: #E2E8F0; 
                                margin: 0 0 8px 0;
                                font-size: 1.2rem;
                                display: flex;
                                align-items: center;
                            ">
                                <span style="font-size: 1.3rem; margin-right: 8px;">📝</span>
                                Paper {idx+1}: {title}
                            </h3>
                            <p style="color: #CBD5E1; margin: 0; font-size: 0.95rem;">
                                <strong>Abstract:</strong> {abstract}
                            </p>
                            {f"<p style='color: #CBD5E1; margin: 8px 0 0 0; font-size: 0.95rem;'><strong>💡 Rationale:</strong> {rationale}</p>" if rationale else ""}
                        </div>
                        """
                        st.markdown(paper_html, unsafe_allow_html=True)
            else:
                st.info("No papers found in this node.")
                
        else:
            st.subheader("Subtopics")
            subtopics, _ = get_subtopics_and_papers(current_node)
            for (subkey, subdict) in subtopics:
                col1, col2, col3 = st.columns([0.5, 3.5, 1])
                with col1:
                    st.write(f"**{subkey}**")
                with col2:
                    sub_subtopics, sub_papers = get_subtopics_and_papers(subdict)
                    st.write(f"- Subtopics: {len(sub_subtopics)}")
                    st.write(f"- Total Papers: {count_papers_recursive(subdict)}")
                with col3:
                    if st.button(f"View '{subkey}'", key=f"btn_{subkey}_{len(st.session_state.path)}"):
                        st.session_state.path.append(subkey)
                        st.rerun()
                st.markdown("---")
                
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
        st.write("Please ensure the JSON file is present and properly formatted.")
        st.write(f"Details: {str(e)}")

if __name__ == '__main__':
    main()
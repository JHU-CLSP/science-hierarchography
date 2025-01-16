import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from urllib.parse import quote, unquote

def get_hierarchy_files():
    hierarchy_dir = 'hierarchies'
    if not os.path.exists(hierarchy_dir):
        return []
    return [f for f in os.listdir(hierarchy_dir) if f.endswith('.json')]

def parse_filename(filename):
    filename = filename.replace('.json', '')
    parts = filename.split('_')
    
    info = {
        'date': parts[0],
        'algorithm': parts[1].replace('Alg:', ''),
        'hyperparams': parts[2].replace('Hyperparams:', ''),
        'paper_count': parts[3].replace('PaperCount:', '')
    }
    
    return info

def format_hierarchy_option(filename):
    info = parse_filename(filename)
    return f"{info['date']} - {info['algorithm']} (k={info['hyperparams'].split('=')[1]}, papers={info['paper_count']})"

def load_hierarchy_data(filename):
    filepath = os.path.join('hierarchies', filename)
    with open(filepath, 'r') as f:
        return json.load(f)

def get_cluster_statistics(clusters):
    def count_papers(node):
        if "children" not in node:
            return 0
        children = node["children"]
        if not children:
            return 0
        if "paper_id" in children[0]:
            return len(children)
        return sum(count_papers(child) for child in children)

    cluster_count = len(clusters)
    paper_counts = []
    
    for cluster, _ in clusters:
        paper_count = count_papers(cluster)
        paper_counts.append(paper_count)
    
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

def find_clusters_in_path(data, path):
    clusters = data["clusters"]
    current_clusters = []
    
    if not path:
        return [(cluster, []) for cluster in clusters]
    
    current = clusters
    for i, p in enumerate(path):
        for cluster in current:
            if cluster["cluster_id"] == p:
                current = cluster["children"]
                if i == len(path) - 1 and current and "paper_id" in current[0]:
                    return [(paper, path) for paper in current]
                else:
                    current_clusters = [(c, path + [c.get("cluster_id", c.get("paper_id"))]) for c in current]
                break
    
    return current_clusters

def display_path_details(path, data):
    if not path:
        return

    st.markdown("### Path Details")
    
    current = data["clusters"]
    levels = {0: 3, 1: 2, 2: 1}
    
    # 创建path details部分
    for i, cluster_id in enumerate(path):
        level = levels.get(i, 1)
        indent = 0 if level == 3 else (32 if level == 2 else 64)
        
        for c in current:
            if c["cluster_id"] == cluster_id:
                st.markdown(f"""
                    <div style='background: white; border-radius: 8px; 
                            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 
                                      0 2px 4px -1px rgba(0, 0, 0, 0.06);
                            border: 1px solid #E5E7EB; padding: 24px; 
                            margin-left: {indent}px;
                            position: relative;'>
                        <div style='display: flex; align-items: center; 
                                justify-content: space-between;
                                margin-bottom: 16px; padding-bottom: 12px; 
                                border-bottom: 1px solid #F3F4F6;'>
                            <div style='display: flex; align-items: center;'>
                                <div style='width: 12px; height: 12px; 
                                        border-radius: 50%; background: #3B82F6; 
                                        margin-right: 8px;'></div>
                                <h4 style='font-size: 1.25rem; font-weight: 600; 
                                        color: #1F2937;'>
                                    Cluster {c["cluster_id"]}
                                </h4>
                            </div>
                            <span style='background-color: #EBF5FF; color: #2563EB; 
                                    padding: 8px 16px; border-radius: 9999px; 
                                    font-weight: 700; font-size: 1.125rem;'>
                                Level {level}
                            </span>
                        </div>
                        <div>
                            <h5 style='font-size: 0.875rem; font-weight: 500; 
                                    color: #6B7280; text-transform: uppercase; 
                                    letter-spacing: 0.05em; margin-bottom: 8px;'>
                                Title
                            </h5>
                            <p style='color: #1F2937; margin-bottom: 16px;'>
                                {c["title"]}
                            </p>
                            <h5 style='font-size: 0.875rem; font-weight: 500; 
                                    color: #6B7280; text-transform: uppercase; 
                                    letter-spacing: 0.05em; margin-bottom: 8px;'>
                                Summary
                            </h5>
                            <p style='color: #374151;'>{c["abstract"]}</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                current = c["children"]
                break

def main():
    st.set_page_config(layout="wide")
    st.title('Paper Cluster Analysis')
    
    try:
        hierarchy_files = get_hierarchy_files()
        
        if not hierarchy_files:
            st.error("No hierarchy files found in /hierarchies directory")
            return
            
        current_url = st.query_params.get('hierarchy', None)
        if current_url:
            current_file = unquote(current_url) + '.json'
        else:
            current_file = None
            
        hierarchy_options = {format_hierarchy_option(f): f for f in hierarchy_files}
        selected_option = st.selectbox(
            'Select Hierarchy',
            options=list(hierarchy_options.keys()),
            index=list(hierarchy_options.values()).index(current_file) if current_file else 0
        )
        
        selected_file = hierarchy_options[selected_option]
        
        if selected_file != current_file:
            url_filename = quote(selected_file.replace('.json', ''))
            st.query_params['hierarchy'] = url_filename
            
        data = load_hierarchy_data(selected_file)
        
        info = parse_filename(selected_file)
        st.markdown("### Hierarchy Information")
        st.markdown(f"- **Date:** {info['date']}")
        st.markdown(f"- **Algorithm:** {info['algorithm']}")
        st.markdown(f"- **Parameters:** {info['hyperparams']}")
        st.markdown(f"- **Total Papers:** {info['paper_count']}")
        st.markdown("---")
        
        if 'path' not in st.session_state:
            st.session_state.path = []
        
        if st.button('← Back') and st.session_state.path:
            st.session_state.path.pop()
            st.rerun()
        
        current_clusters = find_clusters_in_path(data, st.session_state.path)
        current_level = len(st.session_state.path)
        level_name = ['Level 3', 'Level 2', 'Level 1'][current_level] if current_level < 3 else 'Papers'
        
        is_paper_level = current_level >= 3 or "paper_id" in current_clusters[0][0]

        if not is_paper_level:
            st.subheader('Cluster Statistics')
            stats = get_cluster_statistics(current_clusters)
            cols = st.columns(len(stats))
            for col, (key, value) in zip(cols, stats.items()):
                col.metric(label=key, value=value)
        
        if st.session_state.path:
            path_elements = []
            current = data["clusters"]
            for i, cid in enumerate(st.session_state.path):
                level = 3 - i
                for c in current:
                    if c["cluster_id"] == cid:
                        path_elements.append(f"Cluster {cid} (Level {level})")
                        current = c["children"]
                        break
            
            path_str = " > ".join(path_elements)
            st.markdown(f"**Current Path**: {path_str}")

        display_path_details(st.session_state.path, data)
        st.markdown("<div style='margin-bottom: 48px;'></div>", unsafe_allow_html=True)
        st.subheader(f'{level_name}')
        
        for item, full_path in current_clusters:
            col1, col2, col3 = st.columns([0.5, 3.5, 1])
            
            with col1:
                if is_paper_level:
                    st.write(f"**Paper {item.get('paper_id', '')}**")
                else:
                    st.write(f"**Cluster {item['cluster_id']}**")
            
            with col2:
                st.write(f"**Title:** {item['title']}")
                if is_paper_level:
                    st.write(f"**Abstract:** {item['abstract']}")
                else:
                    st.write(f"**Summary:** {item['abstract']}")
            
            with col3:
                if not is_paper_level and "children" in item:
                    count = len(item["children"])
                    next_level_items = item["children"]
                    is_next_level_papers = len(next_level_items) > 0 and "paper_id" in next_level_items[0]
                    
                    display_text = 'View Papers' if is_next_level_papers else f'View Sub-clusters ({count})'
                    
                    if st.button(display_text, key=f"item_{item.get('cluster_id', '')}"):
                        st.session_state.path.append(item['cluster_id'])
                        st.rerun()
            
            st.markdown("---")
        
    except Exception as e:
        st.error(f'Error occurred: {str(e)}')
        st.write('Please ensure all required files are present and have the correct format.')
        st.write(f"Detailed error: {str(e)}")

if __name__ == '__main__':
    main()
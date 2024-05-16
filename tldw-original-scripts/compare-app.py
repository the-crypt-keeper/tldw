import json
import streamlit as st
import glob

def load_analysis_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def display_analysis_data(data):
    tests = data['tests']
    models_list = data['models']
    models = {}
    for idx, model_info in enumerate(models_list):
        models[model_info['id']] = model_info

    # summary table
    summary_cols = st.columns(len(models_list))
    for model_id, model_info in models.items():
        with summary_cols[model_info['idx']]:
            st.subheader(f"{model_info['short_name']}")

    for test_name, test_data in tests.items():
        st.markdown(f"#### {test_name}")

        columns = st.columns(len(models))
        if 'summary' in test_data:
            st.markdown("**Analysis**: "+test_data['summary'])
            
        for model_id, model_result in test_data['results'].items():
            model_info = models[model_id]

            model_result['passing_tests'] = '\n\n'.join([f":blue[{x}]" for x in model_result['passing_tests'].split('\n') if x.strip() != ''])
            model_result['failing_tests'] = '\n\n'.join([f":red[{x}]" for x in model_result['failing_tests'].split('\n') if x.strip() != ''])

            with columns[model_info['idx']]:
                #st.subheader(f"{model_info['short_name']}")
                st.write(model_result['answer'])
                    
st.set_page_config(page_title='Analysis Explorer', layout="wide")
st.markdown("""
        <style>
            .block-container {
                    padding-top: 2rem;
                    padding-bottom: 0rem;
                    padding-left: 3rem;
                    padding-right: 3.5rem;
                }
        </style>
        """, unsafe_allow_html=True)

files = sorted(glob.glob('compare/*.json'))
data = [json.load(open(file,'r')) for file in files]
titles = [x['config']['title'] for x in data]
options = st.selectbox('Select Summary', titles)
idx = titles.index(options)
display_analysis_data(data[idx])

from email_processing import (parse_email, extract_metadata, calculate_relevance_score,
                              extract_entities, perform_topic_clustering, analyze_email_thread,
                              perform_sentiment_analysis, extract_keywords)
from rag_system import embed_emails_batch, retrieve_relevant_emails, generate_response
from typing import List, Dict, Any, BinaryIO
import gradio as gr
import plotly.graph_objs as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
import validators
import re
import sqlite3
from multiprocessing import Pool, cpu_count

DB_PATH = 'email_analysis.db'


def validate_email_file(file: BinaryIO) -> bool:
    try:
        content = file.read().decode('utf-8')
        return 'From:' in content and 'To:' in content and 'Subject:' in content
    except UnicodeDecodeError:
        return False
    finally:
        file.seek(0)


def sanitize_input(input_string: str) -> str:
    return re.sub(r'[^\w\s@.,?!-]', '', input_string)


def process_single_email(file: BinaryIO) -> Dict[str, Any]:
    email_content = file.read().decode('utf-8')
    parsed_email = parse_email(email_content)
    metadata = extract_metadata(parsed_email)
    entities = extract_entities(parsed_email['body'])
    sentiment = perform_sentiment_analysis(parsed_email['body'])
    keywords = extract_keywords(parsed_email['body'])

    return {
        'parsed_email': parsed_email,
        'metadata': metadata,
        'entities': entities,
        'sentiment': sentiment,
        'keywords': keywords
    }


def process_emails(files: List[BinaryIO], query: str) -> Dict[str, Any]:
    with Pool(processes=cpu_count()) as pool:
        all_emails = pool.map(process_single_email, files)

    for email in all_emails:
        email['relevance_score'] = calculate_relevance_score(email['parsed_email']['body'], query)

    all_emails.sort(key=lambda x: x['relevance_score'], reverse=True)

    embed_emails_batch(all_emails)

    email_bodies = [email['parsed_email']['body'] for email in all_emails]
    topics = perform_topic_clustering(email_bodies)
    threads = analyze_email_thread(all_emails)

    # Store emails in SQLite for faster retrieval
    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany('''
            INSERT OR REPLACE INTO emails (email_id, subject, sender, recipient, date, body)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', [(email['parsed_email']['message_id'],
               email['parsed_email']['subject'],
               email['parsed_email']['from'],
               email['parsed_email']['to'],
               email['parsed_email']['date'].isoformat(),
               email['parsed_email']['body']) for email in all_emails])

    return {
        'emails': all_emails,
        'topics': topics,
        'threads': threads
    }


def create_email_analysis_tab():
    with gr.TabItem("Email Analysis"):
        gr.Markdown("# Analyze Multiple Emails")

        with gr.Row():
            file_upload = gr.File(label="Upload Email Files", file_count="multiple")
            query_input = gr.Textbox(label="Enter your query", placeholder="What would you like to know?")
            openai_api_key = gr.Textbox(label="OpenAI API Key", type="password")

        analyze_button = gr.Button("Analyze Emails")

        with gr.Tabs():
            with gr.TabItem("Email List"):
                email_list = gr.Dataframe(
                    headers=["Subject", "From", "To", "Date", "Relevance Score", "Sentiment"],
                    label="Email List"
                )

            with gr.TabItem("Email Content"):
                email_content = gr.Textbox(label="Email Content", lines=10)
                entity_display = gr.JSON(label="Named Entities")
                keyword_display = gr.JSON(label="Keywords")

            with gr.TabItem("Topic Clusters"):
                topic_display = gr.Plot(label="Topic Clusters")

            with gr.TabItem("Sentiment Analysis"):
                sentiment_plot = gr.Plot(label="Sentiment Analysis")

            with gr.TabItem("Email Threads"):
                thread_display = gr.HTML(label="Email Threads")

            with gr.TabItem("RAG Query"):
                rag_query_input = gr.Textbox(label="Ask a question about the emails",
                                             placeholder="Enter your question here")
                rag_query_button = gr.Button("Get Answer")
                rag_response = gr.Textbox(label="Answer", lines=5)

        def analyze_emails(files, query, api_key):
            try:
                query = sanitize_input(query)
                if not query:
                    raise ValueError("Query cannot be empty")

                if not validators.length(api_key, min=20):
                    raise ValueError("Invalid API key")

                result = process_emails(files, query)

                email_data = [
                    [e['parsed_email']['subject'], e['metadata']['from'][0][1],
                     ', '.join([to[1] for to in e['metadata']['to']]),
                     e['parsed_email']['date'].strftime("%Y-%m-%d %H:%M:%S"), f"{e['relevance_score']:.2f}",
                     f"Polarity: {e['sentiment']['polarity']:.2f}, Subjectivity: {e['sentiment']['subjectivity']:.2f}"]
                    for e in result['emails']
                ]

                topic_fig = go.Figure(data=[go.Scatter3d(
                    x=[topic['id'] for topic in result['topics']],
                    y=[float(list(topic['words'].values())[0]) for topic in result['topics']],
                    z=[float(list(topic['words'].values())[1]) for topic in result['topics']],
                    text=[', '.join(topic['words'].keys()) for topic in result['topics']],
                    mode='markers',
                    marker=dict(
                        size=12,
                        color=[topic['id'] for topic in result['topics']],
                        colorscale='Viridis',
                        opacity=0.8
                    )
                )])
                topic_fig.update_layout(title="Topic Clusters")

                sentiments = [e['sentiment'] for e in result['emails']]
                sentiment_fig = go.Figure(data=[go.Scatter(
                    x=[s['polarity'] for s in sentiments],
                    y=[s['subjectivity'] for s in sentiments],
                    mode='markers',
                    text=[e['parsed_email']['subject'] for e in result['emails']],
                    marker=dict(
                        size=10,
                        color=[s['polarity'] for s in sentiments],
                        colorscale='RdYlGn',
                        showscale=True
                    )
                )])
                sentiment_fig.update_layout(
                    title="Sentiment Analysis",
                    xaxis_title="Polarity",
                    yaxis_title="Subjectivity"
                )

                thread_html = "<ul>"
                for thread in result['threads']:
                    thread_html += f"<li>{thread['email']['parsed_email']['subject']}"
                    if thread['replies']:
                        thread_html += "<ul>"
                        for reply in thread['replies']:
                            thread_html += f"<li>{reply['parsed_email']['subject']}</li>"
                        thread_html += "</ul>"
                    thread_html += "</li>"
                thread_html += "</ul>"

                return (
                    email_data,
                    result['emails'][0]['parsed_email']['body'] if result['emails'] else "",
                    result['emails'][0]['entities'] if result['emails'] else {},
                    result['emails'][0]['keywords'] if result['emails'] else [],
                    topic_fig,
                    sentiment_fig,
                    thread_html
                )
            except Exception as e:
                return str(e), "", {}, [], None, None, ""

        def rag_query(question, api_key):
            try:
                question = sanitize_input(question)
                if not question:
                    raise ValueError("Question cannot be empty")

                if not validators.length(api_key, min=20):
                    raise ValueError("Invalid API key")

                relevant_email_ids = retrieve_relevant_emails(question)
                response = generate_response(question, relevant_email_ids, api_key)
                return response
            except Exception as e:
                return str(e)

        analyze_button.click(
            analyze_emails,
            inputs=[file_upload, query_input, openai_api_key],
            outputs=[email_list, email_content, entity_display, keyword_display, topic_display, sentiment_plot,
                     thread_display]
        )

        rag_query_button.click(
            rag_query,
            inputs=[rag_query_input, openai_api_key],
            outputs=[rag_response]
        )
    def display_email_content(evt: gr.SelectData):
        try:
            with sqlite3.connect(DB_PATH) as conn:
                email = conn.execute('''
                    SELECT subject, sender, recipient, date, body
                    FROM emails
                    WHERE rowid = ?
                ''', (evt.index[0] + 1,)).fetchone()

            if not email:
                raise ValueError("Email not found")

            subject, sender, recipient, date, body = email

            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(body)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)

            wordcloud_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            entities = extract_entities(body)
            keywords = extract_keywords(body)

            return (
                body,
                entities,
                keywords,
                f'<img src="data:image/png;base64,{wordcloud_b64}" alt="Word Cloud">'
            )
        except Exception as e:
            return str(e), {}, [], ""


    email_list.select(display_email_content, outputs=[email_content, entity_display, keyword_display, gr.HTML()])

    return file_upload, query_input, openai_api_key, analyze_button, email_list, email_content, entity_display, keyword_display, topic_display, sentiment_plot, thread_display, rag_query_input, rag_query_button, rag_response

# You would typically call create_email_analysis_tab() from your main Gradio interface setup
    import json
import os
import re
from openai import OpenAI
import streamlit as st

# --- Configuration ---
st.set_page_config(layout="wide", page_title="QRM AI Teaching Assistant")

JSON_FILE = "qrm_chapter_summaries_complete.json"
API_KEY = "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"

# --- OpenAI Client ---
try:
    client = OpenAI(api_key=API_KEY)
except Exception as e:
    st.warning(f"Failed to initialize OpenAI client: {e}. Q&A feature will be disabled.")
    client = None

# --- Load Knowledge Base ---
@st.cache_data
def load_knowledge_base(file_path):
    """
    Load and transform QRM JSON data into a format compatible with the chatbot.
    The QRM data is structured as a list of chapters, each containing a list of topics.
    We transform it into a dictionary where keys are chapter_X and values are dictionaries of topics.
    """
    if not os.path.exists(file_path):
        return None, f"Error: Knowledge base file not found at {file_path}"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Transform the data structure
        transformed_data = {}
        
        for chapter in data:
            chapter_number = chapter.get("chapter_number")
            chapter_key = f"chapter_{chapter_number}"
            chapter_topics = {}
            
            for topic in chapter.get("topics", []):
                topic_title = topic.get("topic_title", "")
                if topic_title:
                    chapter_topics[topic_title] = {
                        "summary": topic.get("summary", ""),
                        "subtitles": topic.get("subtitles", []),
                        "quiz_questions": topic.get("quiz_questions", []),
                        "training_tasks": topic.get("training_tasks", [])
                    }
            
            if chapter_topics:  # Only add chapters with topics
                transformed_data[chapter_key] = chapter_topics
        
        # Sort chapters by number
        sorted_chapter_keys = sorted(
            transformed_data.keys(),
            key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else float('inf')
        )
        
        sorted_data = {ch_key: transformed_data[ch_key] for ch_key in sorted_chapter_keys}
        return sorted_data, None
        
    except json.JSONDecodeError:
        return None, f"Error: Could not decode JSON from {file_path}"
    except Exception as e:
        return None, f"Error loading knowledge base: {e}"

knowledge_base, load_error = load_knowledge_base(JSON_FILE)

# --- Helper Function for OpenAI Call ---
def get_ai_response(question, context):
    if not client:
        return "Error: OpenAI client not initialized or API key invalid."
    try:
        prompt = f"""
You are an AI Teaching Assistant for Quantitative Risk Management (QRM).
Based *only* on the following context about the topic, answer the user's question concisely.
If the answer cannot be found in the context, state that clearly.

Context:
{context}

User Question: {question}

Answer:
"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant specializing in Quantitative Risk Management, answering questions based *only* on the provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI API: {e}"

# --- URL Parameter Handling ---
def get_topic_from_url(kb):
    all_info_values = st.query_params.get_all("info")
    param_to_process = st.query_params.get("info")
    
    target_chapter = None
    target_topic = None
    error_message = None
    
    if param_to_process:
        m = re.match(r"chapter(\d+)[-_](.+)", param_to_process, re.IGNORECASE)
        if m:
            chap_no = m.group(1)
            url_topic_segment = m.group(2)
            slug_for_matching = url_topic_segment.replace("_", " ").replace("-", " ")
            chap_key = f"chapter_{chap_no}"
            
            if kb and chap_key in kb:
                for t_key_in_kb in kb[chap_key]:
                    if t_key_in_kb.lower() == slug_for_matching.lower():
                        target_chapter = chap_key
                        target_topic = t_key_in_kb
                        break
                if not target_topic:
                    error_message = f"Topic '{slug_for_matching}' (from URL '{url_topic_segment}') not found in Chapter {chap_no}."
            else:
                error_message = f"Chapter {chap_no} (key '{chap_key}') not found."
        else:
            error_message = (
                f"Invalid format: '{param_to_process}'. "
                "Use chapter<Number>-<TopicName> or chapter<Number>_<TopicName>."
            )
    
    return target_chapter, target_topic, error_message

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_topic_context" not in st.session_state:
    st.session_state.current_topic_context = ""
if "selected_chapter_key" not in st.session_state:
    st.session_state.selected_chapter_key = None
if "selected_topic_name" not in st.session_state:
    st.session_state.selected_topic_name = None
if "_previous_topic" not in st.session_state:
    st.session_state._previous_topic = None

# --- Main App Logic ---
st.title("QRM AI Teaching Assistant")

if load_error:
    st.error(load_error)
    st.stop()

if not knowledge_base:
    st.warning("Knowledge base is empty or could not be loaded.")
    st.stop()

# Get target chapter/topic from URL
url_target_chapter_key, url_target_topic_name, url_parse_error = get_topic_from_url(knowledge_base)

if url_parse_error:
    st.warning(url_parse_error)

# --- Determine Initial Selections for Widgets ---
chapter_options_keys = list(knowledge_base.keys())
chapter_options_display = [f"Chapter {key.split('_')[1]}" for key in chapter_options_keys]

# Determine initial chapter
initial_chapter_key_to_select = None
initial_chapter_index_to_select = 0  # Default to first chapter

if url_target_chapter_key and url_target_chapter_key in chapter_options_keys:
    initial_chapter_key_to_select = url_target_chapter_key
elif st.session_state.selected_chapter_key and st.session_state.selected_chapter_key in chapter_options_keys:
    initial_chapter_key_to_select = st.session_state.selected_chapter_key
elif chapter_options_keys:  # If no URL/session, but chapters exist
    initial_chapter_key_to_select = chapter_options_keys[0]

if initial_chapter_key_to_select:
    initial_chapter_index_to_select = chapter_options_keys.index(initial_chapter_key_to_select)
else:  # Should not happen if knowledge_base has chapters, but as a fallback
    st.warning("No chapters available for selection.")
    st.stop()

# --- Sidebar for Navigation ---
st.sidebar.title("Navigation")

# Chapter Selectbox
selected_chapter_idx = st.sidebar.selectbox(
    "Select Chapter:",
    range(len(chapter_options_keys)),
    index=initial_chapter_index_to_select,
    format_func=lambda i: chapter_options_display[i],
    key="chapter_select_widget"
)
selected_chapter_key = chapter_options_keys[selected_chapter_idx]
st.session_state.selected_chapter_key = selected_chapter_key

# Topic Selectbox
current_chapter_data = knowledge_base.get(selected_chapter_key, {})
topic_options_keys = list(current_chapter_data.keys())
selected_topic_name = None  # Initialize

if not topic_options_keys:
    st.sidebar.warning(f"No topics found for Chapter {selected_chapter_key.split('_')[1]}.")
else:
    initial_topic_name_to_select = None
    initial_topic_index_to_select = 0  # Default to first topic

    # Determine initial topic FOR THE CURRENTLY SELECTED CHAPTER
    if url_target_topic_name and selected_chapter_key == url_target_chapter_key and url_target_topic_name in topic_options_keys:
        initial_topic_name_to_select = url_target_topic_name
    elif st.session_state.selected_topic_name and \
         st.session_state.get("selected_chapter_key_for_topic") == selected_chapter_key and \
         st.session_state.selected_topic_name in topic_options_keys:
        initial_topic_name_to_select = st.session_state.selected_topic_name
    elif topic_options_keys:  # If no URL/session match for this chapter, but topics exist
        initial_topic_name_to_select = topic_options_keys[0]

    if initial_topic_name_to_select:
        initial_topic_index_to_select = topic_options_keys.index(initial_topic_name_to_select)

    selected_topic_idx = st.sidebar.selectbox(
        "Select Topic:",
        range(len(topic_options_keys)),
        index=initial_topic_index_to_select,
        format_func=lambda i: topic_options_keys[i],
        key="topic_select_widget"
    )
    selected_topic_name = topic_options_keys[selected_topic_idx]
    st.session_state.selected_topic_name = selected_topic_name
    st.session_state.selected_chapter_key_for_topic = selected_chapter_key

# --- Main Area ---
if selected_chapter_key and selected_topic_name:
    st.header(f"Chapter {selected_chapter_key.split('_')[1]}: {selected_topic_name}")

    topic_data = knowledge_base[selected_chapter_key][selected_topic_name]

    # Format quiz questions for context
    quiz_context = ""
    quiz_questions = topic_data.get("quiz_questions", [])
    if quiz_questions:
        quiz_context = "Quiz Questions:\n"
        for i, q in enumerate(quiz_questions):
            if isinstance(q, dict):
                question_text = q.get("question", "")
                options = q.get("options", [])
                answer = q.get("answer", "")
                
                quiz_context += f"{i+1}. {question_text}\n"
                if options:
                    for opt in options:
                        quiz_context += f"   - {opt}\n"
                if answer:
                    quiz_context += f"   Answer: {answer}\n"
            else:
                quiz_context += f"{i+1}. {q}\n"
    
    # Format training tasks for context
    tasks_context = ""
    training_tasks = topic_data.get("training_tasks", [])
    if training_tasks:
        tasks_context = "Training Tasks:\n"
        for i, t in enumerate(training_tasks):
            if isinstance(t, dict):
                task_text = t.get("task", "")
                guidance = t.get("guidance", "")
                
                tasks_context += f"{i+1}. {task_text}\n"
                if guidance:
                    tasks_context += f"   Guidance: {guidance}\n"
            else:
                tasks_context += f"{i+1}. {t}\n"

    # Create comprehensive context for the AI
    new_context = (
        f"Topic: {selected_topic_name}\n"
        f"Summary: {topic_data.get('summary', '')}\n"
        f"Subtitles: {', '.join(topic_data.get('subtitles', []))}\n"
        f"{quiz_context}\n"
        f"{tasks_context}"
    )
    
    if st.session_state.current_topic_context != new_context or st.session_state._previous_topic != selected_topic_name:
        st.session_state.current_topic_context = new_context
        st.session_state.messages = []
        st.session_state._previous_topic = selected_topic_name
        st.rerun()

    # Expand details if this specific topic was loaded via URL
    expand_details_default = False
    if url_target_chapter_key == selected_chapter_key and \
       url_target_topic_name == selected_topic_name and \
       not st.session_state.get("_url_params_processed", False):
        expand_details_default = True
    if selected_chapter_key or selected_topic_name:  # After any selection, mark as processed
        st.session_state._url_params_processed = True

    with st.expander("View Topic Details (Summary, Subtitles, Quiz, Tasks)", expanded=expand_details_default):
        st.subheader("Summary")
        st.write(topic_data.get("summary", "Summary not available."))

        st.subheader("Subtitles")
        subtitles = topic_data.get("subtitles", [])
        if subtitles:
            for sub in subtitles:
                st.markdown(f"- {sub}")
        else:
            st.write("No specific subtitles listed for this topic.")

        st.subheader("Quiz Questions")
        if quiz_questions:
            for i, q in enumerate(quiz_questions):
                if isinstance(q, dict):
                    st.markdown(f"**{i+1}. {q.get('question', '')}**")
                    options = q.get("options", [])
                    if options:
                        for opt in options:
                            st.markdown(f"- {opt}")
                    answer = q.get("answer", "")
                    if answer:
                        st.markdown(f"**Answer:** {answer}")
                else:
                    st.markdown(f"{i+1}. {q}")
        else:
            st.write("No quiz questions available.")

        st.subheader("Training Tasks")
        if training_tasks:
            for i, t in enumerate(training_tasks):
                if isinstance(t, dict):
                    st.markdown(f"**{i+1}. {t.get('task', '')}**")
                    guidance = t.get("guidance", "")
                    if guidance:
                        st.markdown(f"*Guidance:* {guidance}")
                else:
                    st.markdown(f"{i+1}. {t}")
        else:
            st.write("No training tasks available.")

    st.divider()
    st.subheader(f"Chat about: {selected_topic_name}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(f"Ask a question about {selected_topic_name}..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = get_ai_response(prompt, st.session_state.current_topic_context)
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.info("Please select a chapter and topic from the sidebar to view content and chat.")


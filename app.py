import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import string
import time
import re

# Explicitly download NLTK resources at the start of application
try:
    # Force download these resources
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    st.error(f"Failed to download NLTK resources: {str(e)}")

# Load stopwords after ensuring download
try:
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words('english'))
except Exception as e:
    st.error(f"Failed to load stopwords: {str(e)}")
    STOP_WORDS = set()  # Fallback empty set

# Function to load models with caching
@st.cache_resource
def load_models():
    with st.spinner("Loading GPT-2 model... This may take a moment."):
        try:
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2LMHeadModel.from_pretrained('gpt2')
            return tokenizer, model
        except Exception as e:
            st.error(f"Failed to load models: {str(e)}")
            return None, None

# Simple tokenization without relying on NLTK word_tokenize
def simple_tokenize(text):
    # Remove punctuation and split by whitespace
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return text.split()

# Advanced perplexity calculation
def calculate_perplexity(text, tokenizer, model):
    if not text.strip() or tokenizer is None or model is None:
        return 0
    
    try:
        # Handle longer texts by splitting into chunks
        max_length = 512
        chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
        perplexities = []
        
        for chunk in chunks:
            if not chunk.strip():
                continue
                
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = inputs["input_ids"]
            
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
                
            try:
                perplexity = torch.exp(loss).item()
                perplexities.append(perplexity)
            except OverflowError:
                perplexities.append(float('inf'))
        
        if not perplexities:
            return 0
            
        return sum(perplexities) / len(perplexities)
    except Exception as e:
        st.error(f"Perplexity calculation error: {str(e)}")
        return 0

# Enhanced burstiness calculation without nltk.word_tokenize
def calculate_burstiness(text):
    if not text.strip():
        return 0
        
    try:
        # Clean the text and tokenize without nltk.word_tokenize
        text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = simple_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in STOP_WORDS]
        
        if not tokens:
            return 0
            
        # Calculate word frequency
        word_freq = {}
        for token in tokens:
            word_freq[token] = word_freq.get(token, 0) + 1
            
        repeated_count = sum(1 for count in word_freq.values() if count > 1)
        burstiness_score = repeated_count / max(len(word_freq), 1)
        
        return burstiness_score
    except Exception as e:
        st.error(f"Burstiness calculation error: {str(e)}")
        return 0

# Calculate n-gram repetition (another feature for AI text detection)
def calculate_ngram_repetition(text, n=3):
    if not text.strip():
        return 0
    
    try:
        # Simple tokenization    
        tokens = simple_tokenize(text)
        
        if len(tokens) < n:
            return 0
            
        # Create n-grams manually
        n_grams = []
        for i in range(len(tokens) - n + 1):
            n_grams.append(tuple(tokens[i:i+n]))
            
        if not n_grams:
            return 0
            
        unique_ngrams = set(n_grams)
        repetition_ratio = 1 - (len(unique_ngrams) / len(n_grams))
        
        return repetition_ratio
    except Exception as e:
        st.error(f"N-gram calculation error: {str(e)}")
        return 0

# Plot top repeated words with improved visualization
def plot_top_repeat_words(text):
    if not text.strip():
        st.warning("No text to analyze.")
        return
    
    try:
        # Clean and tokenize text
        text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = text.split()
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in STOP_WORDS and token not in string.punctuation and len(token) > 2]
        
        # Count word frequencies
        word_counts = {}
        for token in tokens:
            word_counts[token] = word_counts.get(token, 0) + 1
            
        # Get top words
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        if not top_words:
            st.warning("Not enough meaningful words found for analysis.")
            return
            
        words = [word for word, count in top_words]
        counts = [count for word, count in top_words]
        
        # Create bar chart with improved styling
        fig = px.bar(
            x=words, 
            y=counts, 
            labels={'x': 'Words', 'y': 'Frequency'},
            title='Top 10 Repeated Words'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(240,240,240,0.8)',
            font_family="Arial",
            title_font_size=18
        )
        
        fig.update_traces(marker_color='#1f77b4', marker_line_color='#000000', marker_line_width=1)
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Word plot error: {str(e)}")

# Create sentence complexity histogram
def plot_sentence_complexity(text):
    if not text.strip():
        return
    
    try:
        # Simple sentence splitting using periods, question marks, exclamation points
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            st.warning("No sentences detected.")
            return
            
        # Calculate lengths using simple tokenization
        sentence_lengths = [len(simple_tokenize(sentence)) for sentence in sentences]
        
        if not sentence_lengths:
            return
            
        fig = px.histogram(
            x=sentence_lengths, 
            nbins=10,
            labels={'x': 'Sentence Length (words)', 'y': 'Frequency'},
            title='Sentence Length Distribution'
        )
        
        fig.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(240,240,240,0.8)',
            font_family="Arial"
        )
        
        fig.update_traces(marker_color='#2ca02c', marker_line_color='#000000', marker_line_width=1)
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Sentence complexity plot error: {str(e)}")

# Main app
def main():
    st.set_page_config(
        page_title="GPT Shield: AI Plagiarism Detector",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better appearance
    st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stTextInput, .stTextArea {
        background-color: white;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
        padding: 10px;
    }
    .stButton>button {
        background-color: #4e8cff;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #3d7ae5;
    }
    h1, h2, h3 {
        color: #1f2937;
    }
    .stAlert {
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App header with logo
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 20px;">
        <h1 style="margin: 0;">🛡️ GPT Shield: AI Plagiarism Detector</h1>
    </div>
    <p style="margin-bottom: 30px;">Analyze text to detect AI-generated content with advanced NLP techniques</p>
    """, unsafe_allow_html=True)
    
    # Initialize session state for text area
    if "text_area" not in st.session_state:
        st.session_state.text_area = "This is a sample text. You can replace it with any content you want to analyze for potential AI generation patterns."
    
    # Sidebar for app information and settings
    with st.sidebar:
        st.header("About")
        st.markdown("""
        **GPT Shield** helps identify potentially AI-generated text using:
        
        - **Perplexity**: Measures how "surprised" a language model is by text
        - **Burstiness**: Analyzes word repetition patterns
        - **N-gram repetition**: Detects repeated phrases
        - **Sentence complexity**: Analyzes sentence structure variation
        
        AI-generated text typically has lower perplexity and higher burstiness than human text.
        """)
        
        st.header("Settings")
        threshold_perplexity = st.slider("Perplexity Threshold", 10.0, 100.0, 60.0, 0.1, 
                                        help="Lower values suggest AI generation")
        threshold_burstiness = st.slider("Burstiness Threshold", 0.1, 0.5, 0.3, 0.01,
                                        help="Higher values suggest AI generation")
        threshold_ngram = st.slider("N-gram Repetition Threshold", 0.1, 0.5, 0.3, 0.01,
                                   help="Higher values suggest AI generation")
    
    # Load models
    tokenizer, model = load_models()
    
    # Text input area with example
    st.markdown("### Enter text to analyze")
    text_area = st.text_area("", value=st.session_state.text_area, height=200, key="text_input")
    
    analyze_column, clear_column = st.columns([5, 1])
    with analyze_column:
        analyze_button = st.button("Analyze Text", use_container_width=True)
    with clear_column:
        clear_button = st.button("Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.text_area = ""
        st.rerun()  # Using st.rerun() instead of experimental_rerun
    
    if analyze_button and text_area:
        with st.spinner("Analyzing text... This may take a moment."):
            # Add a progress bar for visual feedback
            progress_bar = st.progress(0)
            
            # Calculate metrics with visual progress updates
            progress_bar.progress(20)
            time.sleep(0.3)  # Simulate processing time
            
            perplexity = calculate_perplexity(text_area, tokenizer, model)
            progress_bar.progress(50)
            time.sleep(0.3)
            
            burstiness_score = calculate_burstiness(text_area)
            progress_bar.progress(70)
            time.sleep(0.2)
            
            ngram_repetition = calculate_ngram_repetition(text_area)
            progress_bar.progress(100)
            time.sleep(0.2)
            
            # Clear progress bar after completion
            progress_bar.empty()
        
        # Create scoring system (0-100)
        ai_probability = min(100, max(0, (
            (50 * (1 - min(perplexity / threshold_perplexity, 1))) + 
            (25 * min(burstiness_score / threshold_burstiness, 1)) + 
            (25 * min(ngram_repetition / threshold_ngram, 1))
        )))
        
        # Display results in three columns
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("### Text Analysis")
            
            # Count sentences without nltk
            sentences = re.split(r'[.!?]+', text_area)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Create gauge chart for AI probability
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=ai_probability,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "AI Probability Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1f77b4"},
                    'steps': [
                        {'range': [0, 30], 'color': "#2ca02c"},
                        {'range': [30, 70], 'color': "#ffa500"},
                        {'range': [70, 100], 'color': "#d62728"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': ai_probability
                    }
                }
            ))
            
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                font_family="Arial"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Final verdict with conditional formatting
            if ai_probability > 70:
                st.error("⚠️ Highly likely AI-generated content")
            elif ai_probability > 30:
                st.warning("⚠️ Potentially contains AI-generated content")
            else:
                st.success("✅ Likely human-written content")
                
            st.info(f"""
            **Text Statistics:**
            - Word count: {len(text_area.split())}
            - Character count: {len(text_area)}
            - Sentence count: {len(sentences)}
            """)
        
        with col2:
            st.markdown("### Detailed Metrics")
            
            # Format metrics with better presentation
            metrics_data = {
                "Metric": ["Perplexity", "Burstiness", "N-gram Repetition"],
                "Value": [f"{perplexity:.2f}", f"{burstiness_score:.2f}", f"{ngram_repetition:.2f}"],
                "Interpretation": [
                    "Lower values suggest AI text" if perplexity < threshold_perplexity else "Consistent with human writing",
                    "Higher values suggest AI text" if burstiness_score > threshold_burstiness else "Consistent with human writing",
                    "Higher values suggest AI text" if ngram_repetition > threshold_ngram else "Consistent with human writing"
                ]
            }
            
            # Create a styled metric table
            for i in range(len(metrics_data["Metric"])):
                col_left, col_right = st.columns([1, 1])
                with col_left:
                    st.markdown(f"**{metrics_data['Metric'][i]}**")
                with col_right:
                    if (metrics_data["Metric"][i] == "Perplexity" and perplexity < threshold_perplexity) or \
                       (metrics_data["Metric"][i] != "Perplexity" and float(metrics_data["Value"][i]) > (threshold_burstiness if "Burstiness" in metrics_data["Metric"][i] else threshold_ngram)):
                        st.markdown(f"<span style='color:red; font-weight:bold'>{metrics_data['Value'][i]}</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span style='color:green; font-weight:bold'>{metrics_data['Value'][i]}</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='font-size:0.9em; color:#505050'>{metrics_data['Interpretation'][i]}</span>", unsafe_allow_html=True)
                st.markdown("---")
            
            st.markdown("""
            ### Understanding the Results
            
            - **Perplexity**: How predictable the text is to the AI model
            - **Burstiness**: Measures word repetition patterns
            - **N-gram Repetition**: Detects repeated phrases
            """)
            
        with col3:
            st.markdown("### Text Patterns")
            
            # Show advanced visualizations
            plot_top_repeat_words(text_area)
            plot_sentence_complexity(text_area)
        
        # Disclaimer
        st.markdown("""
        ---
        **Disclaimer**: This tool provides an assessment based on statistical patterns commonly associated with AI-generated text. 
        The results should be considered indicative rather than definitive. False positives and false negatives can occur. 
        Always use human judgment alongside automated tools for content evaluation.
        """)

if __name__ == "__main__":
    main()
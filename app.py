# app.py

import streamlit as st
import pandas as pd
import pickle
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="SMS Spam Detection System",
    page_icon="📩",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

import joblib

model = joblib.load("spam_classifier.pkl")

# =====================================================
# NLTK
# =====================================================

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

ps = PorterStemmer()

# =====================================================
# TEXT PREPROCESSING
# =====================================================

def transform_text(text):

    text = text.lower()

    words = nltk.word_tokenize(text)

    filtered = []

    for word in words:
        if word.isalnum():
            filtered.append(word)

    words = []

    for word in filtered:
        if word not in stopwords.words('english') and word not in string.punctuation:
            words.append(word)

    stemmed = []

    for word in words:
        stemmed.append(ps.stem(word))

    return " ".join(stemmed)

# =====================================================
# CSS
# =====================================================

st.markdown("""
<style>

/* Background */

.stApp{
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b,
        #334155
    );
}

/* Sidebar */

section[data-testid="stSidebar"]{
    background:#111827;
}

section[data-testid="stSidebar"] *{
    color:white !important;
}

/* Title */

.main-title{
    text-align:center;
    font-size:85px;
    font-weight:900;

    background: linear-gradient(
        90deg,
        #ff0000,
        #ff7300,
        #ffee00,
        #00ff44,
        #00e1ff,
        #0051ff,
        #8f00ff
    );

    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.sub-title{
    text-align:center;
    color:white;
    font-size:26px;
    margin-bottom:20px;
}

/* Text Area */

textarea{
    font-size:28px !important;
    font-weight:600 !important;
    color:black !important;
    background:white !important;
}

textarea::placeholder{
    font-size:22px !important;
}

[data-testid="stTextArea"] textarea{
    border-radius:20px !important;
    border:3px solid #38bdf8 !important;
}

/* Button */

.stButton > button{

    width:100%;
    height:75px;

    border:none;
    border-radius:20px;

    font-size:24px;
    font-weight:bold;

    color:white;

    background:linear-gradient(
        90deg,
        #ff0080,
        #7928ca,
        #0070f3
    );
}

.stButton > button:hover{

    transform:scale(1.02);

    box-shadow:0px 0px 20px #38bdf8;
}

/* Metrics */

[data-testid="stMetric"]{
    background:#1e293b;
    padding:15px;
    border-radius:15px;
    border:1px solid #475569;
}

[data-testid="stMetricValue"]{
    color:white !important;
    font-size:45px !important;
    font-weight:bold !important;
}

[data-testid="stMetricLabel"]{
    color:#38bdf8 !important;
    font-size:20px !important;
}

/* Result Cards */

.spam-card{
    background:#ef4444;
    padding:30px;
    border-radius:20px;
    text-align:center;
    color:white;
    font-size:40px;
    font-weight:bold;
}

.ham-card{
    background:#22c55e;
    padding:30px;
    border-radius:20px;
    text-align:center;
    color:white;
    font-size:40px;
    font-weight:bold;
}

/* Footer */

.footer{
    text-align:center;
    color:white;
    padding:30px;
}

.footer h2{
    font-size:38px;
}

.footer h3{
    font-size:24px;
}

.footer p{
    font-size:18px;
}
            
/* Success Message */

.stSuccess{
    font-size:24px !important;
    font-weight:bold !important;
}

/* Warning Message */

.stWarning{
    font-size:24px !important;
    font-weight:bold !important;
}

/* Error Message */

.stError{
    font-size:24px !important;
    font-weight:bold !important;
}            

</style>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("📊 Project Information")

    st.success("Accuracy: 96.81%")

    st.markdown("""
### 🤖 Model
Multinomial Naive Bayes

### ⚙ Features
- TF-IDF
- Message Length
- Word Count
- Character Count
- Sentence Count

### 🎯 Applications
- SMS Spam Detection
- Email Filtering
- NLP Projects
""")

# =====================================================
# HEADER
# =====================================================

st.markdown(
    "<div class='main-title'>📩 SMS Spam Detection System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>AI Powered Spam Message Classification using NLP & Machine Learning</div>",
    unsafe_allow_html=True
)

st.divider()

# =====================================================
# INPUT
# =====================================================

message = st.text_area(
    "✍ Enter SMS Message",
    height=320,
    placeholder="Type your SMS message here..."
)

# =====================================================
# LIVE METRICS
# =====================================================

if message:

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📝 Characters", len(message))

    with col2:
        st.metric("📖 Words", len(message.split()))

    with col3:
        st.metric("📄 Sentences", len(nltk.sent_tokenize(message)))

# =====================================================
# PREDICT
# =====================================================

if st.button("🚀 Predict Message"):

    transformed_message = transform_text(message)
    sentence_count = len([s for s in message.split(".") if s.strip()])

    input_df = pd.DataFrame({
        "transformed_message":[transformed_message],
        "message_length":[len(message)],
        "word_count":[len(message.split())],
        "char_count":[len(message)],
        "sentence_count": [sentence_count]
    })

    prediction = model.predict(input_df)[0]

    st.divider()

    if prediction == 1:

        st.markdown("""
        <div class='spam-card'>
        🚨 SPAM MESSAGE DETECTED
        </div>
        """, unsafe_allow_html=True)

        st.warning(
            "This message appears suspicious and may contain spam content."
        )

        st.snow()

    else:

        st.markdown("""
        <div class='ham-card'>
        ✅ HAM MESSAGE DETECTED
        </div>
        """, unsafe_allow_html=True)

        st.success(
            "This message appears legitimate and safe."
        )

        st.balloons()


# =====================================================
# LIVE METRICS
# =====================================================

if message:

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "📝 Characters",
            len(message)
        )

    with col2:
        st.metric(
            "📖 Words",
            len(message.split())
        )

    with col3:
        sentence_count = len([s for s in message.split(".") if s.strip()])

        st.metric("📄 Sentences", sentence_count)
    

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# PREDICT BUTTON
# =====================================================

predict = st.button(
    "🚀 Predict Message",
    use_container_width=True
)

# =====================================================
# PREDICTION
# =====================================================

if predict:

    if message.strip() == "":

        st.error("⚠ Please enter a message.")

    else:

        transformed_message = transform_text(message)

        input_df = pd.DataFrame({
            "transformed_message":[transformed_message],
            "message_length":[len(message)],
            "word_count":[len(message.split())],
            "char_count":[len(message)],
            "sentence_count":[len(nltk.sent_tokenize(message))]
        })

        prediction = model.predict(input_df)[0]

        st.divider()

        # ==========================================
        # CONFIDENCE SCORE
        # ==========================================

        try:

            probabilities = model.predict_proba(input_df)[0]

            confidence = max(probabilities) * 100

            st.metric(
                "🎯 Confidence Score",
                f"{confidence:.2f}%"
            )

        except:
            pass

        # ==========================================
        # SPAM RESULT
        # ==========================================

        if prediction == 1:

            st.markdown("""
            <div class='spam-card'>
            🚨 SPAM MESSAGE DETECTED
            </div>
            """,
            unsafe_allow_html=True)

            st.warning(
                """
                This message appears suspicious.

                Avoid clicking unknown links,
                sharing OTPs,
                passwords,
                or banking details.
                """
            )

            st.snow()

        # ==========================================
        # HAM RESULT
        # ==========================================

        else:

            st.markdown("""
            <div class='ham-card'>
            ✅ HAM MESSAGE DETECTED
            </div>
            """,
            unsafe_allow_html=True)

            st.success(
                """
                This message appears legitimate
                and safe.
                """
            )

            st.balloons()

# =====================================================
# PROJECT SUMMARY
# =====================================================

st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model", "NB")

with col2:
    st.metric("Features", "5")

with col3:
    st.metric("TF-IDF", "3000")

with col4:
    st.metric("Accuracy", "96.81%")

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown("""
<div class="footer">

<h1>
🚀 Khaja Mainuddin
</h1>

<h3>
Artificial Intelligence & Data Science Engineer
</h3>

<p>
Machine Learning • NLP • Data Science • Python • Streamlit
</p>

<p>
SMS Spam Detection using TF-IDF and Multinomial Naive Bayes
</p>

</div>
""",
unsafe_allow_html=True)

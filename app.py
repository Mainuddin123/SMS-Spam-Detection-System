import streamlit as st
import pandas as pd
import joblib
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
# DOWNLOAD NLTK
# =====================================================

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("spam_classifier.pkl")

ps = PorterStemmer()

# =====================================================
# TEXT PREPROCESSING
# =====================================================

def transform_text(text):

    text = text.lower()

    words = nltk.word_tokenize(text)

    words = [w for w in words if w.isalnum()]

    words = [
        w for w in words
        if w not in stopwords.words("english")
        and w not in string.punctuation
    ]

    words = [ps.stem(w) for w in words]

    return " ".join(words)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#0f172a,#1e293b,#334155);
}

section[data-testid="stSidebar"]{
background:#111827;
}

section[data-testid="stSidebar"] *{
color:white!important;
}

.main-title{
text-align:center;
font-size:70px;
font-weight:900;
background:linear-gradient(90deg,#ff0000,#ff7300,#ffee00,#00ff44,#00e1ff,#0051ff,#8f00ff);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
}

.sub-title{
text-align:center;
font-size:22px;
color:white;
margin-bottom:20px;
}

[data-testid="stTextArea"] textarea{
background:white!important;
color:black!important;
font-size:22px!important;
font-weight:600;
border-radius:18px!important;
border:3px solid #38bdf8!important;
}

.stButton>button{
width:100%;
height:70px;
border:none;
border-radius:18px;
font-size:22px;
font-weight:bold;
color:white;
background:linear-gradient(90deg,#ff0080,#7928ca,#0070f3);
}

[data-testid="stMetric"]{
background:#263247;
padding:18px;
border-radius:15px;
border:2px solid #3b82f6;
}

[data-testid="stMetricLabel"]{
color:#9cc9ff!important;
font-weight:bold;
}

[data-testid="stMetricValue"]{
color:white!important;
font-size:32px!important;
font-weight:bold;
}

.spam-card{
background:#ef4444;
padding:25px;
border-radius:18px;
text-align:center;
color:white;
font-size:34px;
font-weight:bold;
}

.ham-card{
background:#22c55e;
padding:25px;
border-radius:18px;
text-align:center;
color:white;
font-size:34px;
font-weight:bold;
}

.footer{
text-align:center;
color:white;
padding:20px;
}

</style>
""",unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("📊 Project Information")

    st.success("Accuracy : 96.81%")

    st.markdown("""
### 🤖 Model
Multinomial Naive Bayes

### ⚙ Features
- TF-IDF
- Character Count
- Word Count
- Sentence Count

### 🎯 Applications
- SMS Spam Detection
- Email Filtering
- NLP
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
height=260,
placeholder="Type your SMS here..."
)

sentence_count = len([s for s in message.split(".") if s.strip()])

if message:

    c1,c2,c3 = st.columns(3)

    with c1:
        st.metric("📝 Characters",len(message))

    with c2:
        st.metric("📖 Words",len(message.split()))

    with c3:
        st.metric("📄 Sentences",sentence_count)

st.markdown("<br>",unsafe_allow_html=True)

# =====================================================
# PREDICTION
# =====================================================

if st.button("🚀 Predict Message", use_container_width=True):

    if message.strip() == "":
        st.error("⚠ Please enter an SMS message.")
        st.stop()

    transformed_message = transform_text(message)

    input_df = pd.DataFrame({
        "transformed_message": [transformed_message],
        "message_length": [len(message)],
        "word_count": [len(message.split())],
        "char_count": [len(message)],
        "sentence_count": [sentence_count]
    })

    prediction = model.predict(input_df)[0]

    st.divider()

    # =====================================================
    # CONFIDENCE SCORE
    # =====================================================

    try:
        probabilities = model.predict_proba(input_df)[0]
        confidence = max(probabilities) * 100

        st.markdown("## 🎯 Confidence Score")

        st.progress(confidence / 100)

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    except Exception:
        st.info("Confidence score is not available for this model.")

    # =====================================================
    # RESULT
    # =====================================================

    if prediction == 1:

        st.markdown("""
        <div class='spam-card'>
        🚨 SPAM MESSAGE DETECTED
        </div>
        """, unsafe_allow_html=True)

        st.error("""
⚠ This message appears suspicious.

• Avoid clicking unknown links.

• Never share OTPs or passwords.

• Do not disclose banking information.
""")

    else:

        st.markdown("""
        <div class='ham-card'>
        ✅ HAM MESSAGE DETECTED
        </div>
        """, unsafe_allow_html=True)

        st.success("""
This message appears genuine and safe.

No suspicious content detected.
""")

# =====================================================
# PROJECT SUMMARY
# =====================================================

st.divider()

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("🤖 Model", "Multinomial NB")

with m2:
    st.metric("📊 TF-IDF", "3000")

with m3:
    st.metric("⚙ Features", "5")

with m4:
    st.metric("🎯 Accuracy", "96.81%")

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown("""
<div class="footer">

<h2>🚀 Developed by Shaik Khaja Mainuddin</h2>

<h3>Artificial Intelligence & Data Science Engineer</h3>

<p>
Machine Learning • NLP • Python • Streamlit
</p>

<p>
SMS Spam Detection using TF-IDF & Multinomial Naive Bayes
</p>

</div>
""", unsafe_allow_html=True)

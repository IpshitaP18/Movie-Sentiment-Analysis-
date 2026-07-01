import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import preprocess_text

st.set_page_config(
    page_title="The Ticket Booth — Sentiment Cinema",
    page_icon="🎟️",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,200..800&family=Dosis:wght@200..800&family=Josefin+Sans:ital,wght@0,100..700;1,100..700&family=Nunito:ital,wght@0,200..1000;1,200..1000&family=Oswald:wght@200..700&display=swap');

    :root {
        --house-black: #0e0d0f;
        --house-charcoal: #18161a;
        --house-card: #1f1c21;
        --marquee-gold: #d9a635;
        --marquee-gold-dim: #8a6d2a;
        --velvet-red: #a02334;
        --velvet-red-glow: rgba(160,35,52,0.18);
        --emerald: #2f8f5b;
        --emerald-glow: rgba(47,143,91,0.18);
        --paper: #ece6d6;
        --paper-dim: #b6ae9a;
    }

    .stApp {
        background:
            radial-gradient(ellipse at top, #201c22 0%, var(--house-black) 60%);
        color: var(--paper);
    }

    /* Kill default streamlit padding weirdness at top */
    .block-container { padding-top: 1.5rem; }

    h1, h2, h3 {
        font-family: 'Oswald', sans-serif !important;
        letter-spacing: 1.5px;
        color: var(--marquee-gold) !important;
    }

    p, span, label, div, li { font-family: 'Nunito', sans-serif; }

    code, .mono { font-family: 'Dosis', sans-serif !important; font-weight: 600; }

    /* ---------- Sprocket-hole divider (signature motif) ---------- */
    .sprocket-strip {
        height: 22px;
        margin: 6px 0 22px 0;
        background-image:
            radial-gradient(circle 6px, var(--house-black) 6px, transparent 7px);
        background-color: var(--marquee-gold-dim);
        background-repeat: repeat-x;
        background-size: 34px 22px;
        background-position: center;
        border-radius: 3px;
        opacity: 0.85;
    }

    /* ---------- Marquee header ---------- */
    .marquee-box {
        background: linear-gradient(180deg, #211d24 0%, #16131a 100%);
        border: 2px solid var(--marquee-gold-dim);
        border-radius: 6px;
        padding: 22px 30px 16px 30px;
        margin-bottom: 4px;
        position: relative;
        box-shadow: 0 0 40px rgba(217,166,53,0.06), inset 0 0 30px rgba(0,0,0,0.4);
    }
    .marquee-bulbs {
        display: flex;
        justify-content: space-between;
        padding: 0 4px 10px 4px;
    }
    .bulb {
        width: 7px; height: 7px; border-radius: 50%;
        background: var(--marquee-gold);
        box-shadow: 0 0 6px 2px rgba(217,166,53,0.7);
        animation: flicker 2.4s infinite ease-in-out;
    }
    .bulb:nth-child(odd) { animation-delay: 0.6s; }
    .bulb:nth-child(3n) { animation-delay: 1.2s; }
    @keyframes flicker { 0%,100% {opacity:1;} 50% {opacity:0.35;} }

    .marquee-title { font-family: 'Bricolage Grotesque', sans-serif !important; font-weight: 800; font-size: 46px; margin: 0; line-height: 1; }
    .marquee-sub { color: var(--paper-dim); font-size: 14px; margin-top: 6px; letter-spacing: 0.5px; }

    /* ---------- Generic house card ---------- */
    .house-card {
        background: var(--house-card);
        border: 1px solid rgba(217,166,53,0.15);
        border-radius: 10px;
        padding: 22px;
        margin-bottom: 18px;
    }

    /* ---------- Ticket stub verdict (SIGNATURE ELEMENT) ---------- */
    .ticket {
        display: flex;
        border-radius: 8px;
        overflow: hidden;
        margin-top: 6px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }
    .ticket-main {
        flex: 1;
        padding: 22px 24px;
        position: relative;
    }
    .ticket-stub {
        width: 110px;
        display: flex;
        align-items: center;
        justify-content: center;
        writing-mode: vertical-rl;
        font-family: 'Oswald', sans-serif;
        font-size: 20px;
        letter-spacing: 4px;
    }
    /* perforation between stub and body */
    .ticket-main { border-right: 3px dashed rgba(20,18,20,0.35); }
    .ticket-main::before, .ticket-main::after {
        content: "";
        position: absolute;
        right: -13px;
        width: 22px; height: 22px;
        background: var(--house-black);
        border-radius: 50%;
    }
    .ticket-main::before { top: -11px; }
    .ticket-main::after { bottom: -11px; }

    .ticket.positive .ticket-main { background: var(--emerald-glow); }
    .ticket.positive .ticket-stub { background: var(--emerald); color: #eafff2; }
    .ticket.negative .ticket-main { background: var(--velvet-red-glow); }
    .ticket.negative .ticket-stub { background: var(--velvet-red); color: #ffecee; }

    .ticket-code { color: var(--paper-dim); font-size: 11px; letter-spacing: 1px; font-family: 'Dosis', sans-serif; font-weight: 600; }
    .ticket-verdict { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 30px; margin: 4px 0 6px 0; }
    .ticket.positive .ticket-verdict { color: #6fe0a0; }
    .ticket.negative .ticket-verdict { color: #ff8a94; }
    .ticket-desc { font-family: 'Josefin Sans', sans-serif; font-size: 14.5px; color: var(--paper); opacity: 0.9; }

    /* ---------- Confidence meter ---------- */
    .conf-track {
        width: 100%; height: 10px; border-radius: 6px;
        background: rgba(255,255,255,0.08);
        margin-top: 14px; overflow: hidden;
    }
    .conf-fill { height: 100%; border-radius: 6px; }
    .conf-label { font-size: 11px; color: var(--paper-dim); margin-top: 6px; letter-spacing: 0.5px; }

    /* ---------- Preset cards ---------- */
    .preset-card {
        background: var(--house-card);
        border: 1px solid rgba(217,166,53,0.12);
        border-left: 4px solid var(--marquee-gold-dim);
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .preset-tag { font-size: 10px; letter-spacing: 1.5px; color: var(--marquee-gold); text-transform: uppercase; font-family: 'Dosis', sans-serif; font-weight: 600; }
    .preset-text { font-family: 'Josefin Sans', sans-serif; font-style: italic; font-size: 14px; color: var(--paper-dim); margin-top: 4px; }

    /* ---------- Filmstrip history reel ---------- */
    .reel-wrap { display: flex; gap: 8px; overflow-x: auto; padding: 10px 4px 14px 4px; }
    .frame {
        min-width: 150px; max-width: 150px;
        background: var(--house-charcoal);
        border-radius: 4px;
        padding: 8px 10px;
        border-top: 8px solid var(--house-black);
        border-bottom: 8px solid var(--house-black);
        background-image:
            radial-gradient(circle 3px, var(--house-black) 3px, transparent 4px),
            radial-gradient(circle 3px, var(--house-black) 3px, transparent 4px);
        background-repeat: repeat-x, repeat-x;
        background-position: top left, bottom left;
        background-size: 18px 8px, 18px 8px;
    }
    .frame.pos { border-left: 3px solid var(--emerald); }
    .frame.neg { border-left: 3px solid var(--velvet-red); }
    .frame-verdict { font-size: 11px; font-weight: 700; letter-spacing: 0.5px; }
    .frame.pos .frame-verdict { color: #6fe0a0; }
    .frame.neg .frame-verdict { color: #ff8a94; }
    .frame-text { font-size: 11px; color: var(--paper-dim); margin-top: 4px; display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #17141a 0%, #0e0d0f 100%) !important;
        border-right: 1px solid rgba(217,166,53,0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown p { color: var(--paper-dim); }

    /* ---------- Tabs ---------- */
    button[data-baseweb="tab"] {
        font-family: 'Oswald', sans-serif !important;
        font-size: 15px !important;
        letter-spacing: 1px;
        color: var(--paper-dim) !important;
        background-color: transparent !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--marquee-gold) !important;
        border-bottom-color: var(--marquee-gold) !important;
    }

    /* ---------- Inputs / buttons ---------- */
    .stTextArea textarea {
        background-color: #131117 !important;
        color: var(--paper) !important;
        border: 1px solid rgba(217,166,53,0.25) !important;
        border-radius: 6px !important;
    }
    .stTextArea textarea::placeholder {
        color: #6f8fae !important;
        font-family: 'Josefin Sans', sans-serif;
        font-style: italic;
        opacity: 0.85 !important;
    }
    .stButton button {
        background-color: var(--marquee-gold) !important;
        color: #16131a !important;
        font-family: 'Oswald', sans-serif !important;
        letter-spacing: 1px;
        border: none !important;
        border-radius: 6px !important;
    }
    .stButton button:hover { background-color: #f0c25a !important; }

    /* ---------- Native st.container(border=True) styled as a house-card ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--house-card);
        border: 1px solid rgba(217,166,53,0.15) !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    with open("models/sentiment_model.pkl", "rb") as file:
        model = pickle.load(file)
    with open("models/tfidf.pkl", "rb") as file:
        tfidf = pickle.load(file)
    return model, tfidf

@st.cache_data
def load_data():
    return pd.read_csv("dataset/cleaned_IMDB_Dataset.csv")

model, tfidf = load_models()

if "reel" not in st.session_state:
    st.session_state.reel = [] 

TICKET_COUNTER_START = 4021  


def get_confidence(vector):
    """Return probability (0-1) of the predicted class, if the model supports it."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vector)[0]
        return float(np.max(proba))
    if hasattr(model, "decision_function"):
        score = model.decision_function(vector)[0]
        return float(1 / (1 + np.exp(-abs(score))))
    return None


def render_ticket(review_text, prediction, confidence, ticket_no):
    """
    Builds the ticket-stub verdict card as a single flat HTML string
    (no embedded newlines/indentation) so Streamlit's Markdown parser
    never mistakes any part of it for a 4-space-indented code block.
    """
    is_pos = prediction == "positive"
    cls = "positive" if is_pos else "negative"
    verdict_word = "ADMIT" if is_pos else "DENIED"
    desc = ("Bright, favorable language throughout — this one's a crowd-pleaser."
            if is_pos else
            "Language leans critical or dissatisfied — a tough house tonight.")

    conf_html = ""
    if confidence is not None:
        pct = round(confidence * 100)
        bar_color = "var(--emerald)" if is_pos else "var(--velvet-red)"
        conf_html = (
            f"<div class='conf-track'><div class='conf-fill' style='width:{pct}%; background:{bar_color};'></div></div>"
            f"<div class='conf-label mono'>MODEL CONFIDENCE — {pct}%</div>"
        )

    verdict_emoji = "🎉" if is_pos else "🚨"
    verdict_label = "POSITIVE" if is_pos else "NEGATIVE"

    ticket_html = (
        f"<div class='ticket {cls}'>"
        f"<div class='ticket-main'>"
        f"<div class='ticket-code'>TICKET NO. {ticket_no} · SCREENING ROOM 1 · {len(review_text)} CHARS SCANNED</div>"
        f"<div class='ticket-verdict'>{verdict_emoji} VERDICT: {verdict_label}</div>"
        f"<div class='ticket-desc'>{desc}</div>"
        f"{conf_html}"
        f"</div>"
        f"<div class='ticket-stub'>{verdict_word}</div>"
        f"</div>"
    )
    st.markdown(ticket_html, unsafe_allow_html=True)


def log_to_reel(review_text, prediction):
    st.session_state.reel.insert(0, {
        "text": review_text[:60] + ("…" if len(review_text) > 60 else ""),
        "verdict": prediction
    })
    st.session_state.reel = st.session_state.reel[:10]

with st.sidebar:
    st.markdown("<h2 style='text-align:center;'>🎞️ PROJECTOR ROOM</h2>", unsafe_allow_html=True)
    st.markdown("<div class='sprocket-strip'></div>", unsafe_allow_html=True)

    sidebar_card_html = (
        "<div class='house-card'>"
        "<p class='preset-tag'>ENGINE</p>"
        "<p style='color:var(--paper); font-size:14px; margin-bottom:14px;'>NLP · TF-IDF Vectorizer</p>"
        "<p class='preset-tag'>REEL SOURCE</p>"
        "<p style='color:var(--paper); font-size:14px; margin-bottom:14px;'>IMDB Dataset</p>"
        "<p class='preset-tag'>SHOWING</p>"
        "<p style='color:var(--paper); font-size:14px; margin-bottom:0;'>Binary Sentiment Split</p>"
        "</div>"
    )
    st.markdown(sidebar_card_html, unsafe_allow_html=True)

    st.markdown("<div class='sprocket-strip'></div>", unsafe_allow_html=True)
    st.markdown("<p class='preset-tag'>TONIGHT'S REEL</p>", unsafe_allow_html=True)
    if st.session_state.reel:
        for frame in st.session_state.reel[:5]:
            dot = "🟢" if frame["verdict"] == "positive" else "🔴"
            frame_line_html = f"<p style='font-size:12px; color:var(--paper-dim); margin:2px 0;'>{dot} {frame['text']}</p>"
            st.markdown(frame_line_html, unsafe_allow_html=True)
    else:
        st.markdown("<p style='font-size:12px; color:var(--paper-dim);'>No screenings yet tonight.</p>", unsafe_allow_html=True)

bulbs_html = "".join(["<div class='bulb'></div>" for _ in range(22)])
marquee_html = (
    "<div class='marquee-box'>"
    f"<div class='marquee-bulbs'>{bulbs_html}</div>"
    "<h1 class='marquee-title'>THE TICKET BOOTH</h1>"
    "<p class='marquee-sub'>NOW SCREENING · AI SENTIMENT ANALYSIS FOR MOVIE REVIEWS</p>"
    "</div>"
)
st.markdown(marquee_html, unsafe_allow_html=True)
st.markdown("<div class='sprocket-strip'></div>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎫 Box Office Booth", "📽️ Screening Numbers", "🎬 Behind the Scenes"])

with tab1:
    col1, col2 = st.columns([1.8, 1.2], gap="large")

    with col1:
        with st.container(border=True):
            st.markdown("#### 🎟️ Submit a Review for Review")
            review = st.text_area(
                "Drop your raw review transcript here:",
                placeholder="Type something glowing or scathing...",
                height=150,
                label_visibility="collapsed"
            )
            predict_clicked = st.button("PUNCH THE TICKET", type="primary", width="stretch")

        if predict_clicked:
            if not review.strip():
                st.warning("The booth needs a review before it can issue a ticket.")
            else:
                with st.spinner("Running the reel through the projector..."):
                    clean_review = preprocess_text(review)
                    vector = tfidf.transform([clean_review])
                    prediction = model.predict(vector)[0]
                    confidence = get_confidence(vector)
                ticket_no = TICKET_COUNTER_START + len(st.session_state.reel)
                render_ticket(review, prediction, confidence, ticket_no)
                if prediction == "positive":
                    st.balloons()
                log_to_reel(review, prediction)

    with col2:
        st.markdown("#### 🎬 Preset Screenings")
        examples = {
            "Positive Probe": "This movie was absolutely amazing. The story was excellent and the actors performed brilliantly.",
            "Negative Probe": "This movie was boring and disappointing. The acting was terrible and the story was very weak."
        }
        for title, example_review in examples.items():
            preset_card_html = (
                "<div class='preset-card'>"
                f"<span class='preset-tag'>{title}</span>"
                f"<p class='preset-text'>\"{example_review}\"</p>"
                "</div>"
            )
            st.markdown(preset_card_html, unsafe_allow_html=True)
            if st.button(f"Screen {title.split()[0]}", key=title, width="stretch"):
                clean_example = preprocess_text(example_review)
                vector = tfidf.transform([clean_example])
                prediction = model.predict(vector)[0]
                confidence = get_confidence(vector)
                ticket_no = TICKET_COUNTER_START + len(st.session_state.reel)
                render_ticket(example_review, prediction, confidence, ticket_no)
                log_to_reel(example_review, prediction)

    if st.session_state.reel:
        st.markdown("<div class='sprocket-strip'></div>", unsafe_allow_html=True)
        st.markdown("#### 🎞️ Tonight's Filmstrip")
        frame_htmls = []
        for frame in st.session_state.reel:
            cls = "pos" if frame["verdict"] == "positive" else "neg"
            label = "POSITIVE" if frame["verdict"] == "positive" else "NEGATIVE"
            frame_htmls.append(
                f"<div class='frame {cls}'>"
                f"<div class='frame-verdict'>{label}</div>"
                f"<span class='frame-text'>{frame['text']}</span>"
                f"</div>"
            )
        frames_html = f"<div class='reel-wrap'>{''.join(frame_htmls)}</div>"
        st.markdown(frames_html, unsafe_allow_html=True)

with tab2:
    st.markdown("#### 📽️ Box Office Ledger")
    try:
        df = load_data()
        sentiment_count = df["sentiment"].value_counts()

        c1, c2 = st.columns(2)
        with c1:
            total_records_html = (
                "<div class='house-card' style='text-align:center;'>"
                "<p class='preset-tag'>TOTAL RECORDS</p>"
                f"<h2 style='font-size:34px; margin:6px 0;'>{len(df):,}</h2>"
                "</div>"
            )
            st.markdown(total_records_html, unsafe_allow_html=True)
        with c2:
            target_balance_html = (
                "<div class='house-card' style='text-align:center;'>"
                "<p class='preset-tag'>TARGET BALANCE</p>"
                "<h2 style='font-size:34px; margin:6px 0;'>Equal Split</h2>"
                "</div>"
            )
            st.markdown(target_balance_html, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(7, 3.2))
        fig.patch.set_facecolor('#0e0d0f')
        ax.set_facecolor('#0e0d0f')

        colors = ["#2f8f5b", "#a02334"] if sentiment_count.index[0] == 'positive' else ["#a02334", "#2f8f5b"]
        bars = ax.bar(sentiment_count.index.str.upper(), sentiment_count.values, color=colors, width=0.4, edgecolor='none')

        ax.set_ylabel("Review Count", fontsize=9, color="#b6ae9a")
        ax.tick_params(colors='#b6ae9a', labelsize=9)
        ax.grid(axis='y', color='#26232a', linestyle='-')
        for spine in ax.spines.values():
            spine.set_visible(False)

        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval * 0.5, f"{yval:,}",
                     ha='center', va='center', color='#ece6d6', fontweight='bold', fontsize=9)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Could not load the box office ledger: {e}")

with tab3:
    st.markdown("#### 🎬 Confusion Matrix — Director's Cut")
    st.write("Validation topology for the production pipeline configuration.")

    col_img, _ = st.columns([1.5, 1.5])
    with col_img:
        try:
            st.image(
                "images/confusion_matrix.png",
                caption="Production Matrix Array",
                width="stretch"
            )
        except Exception:
            st.warning("⚠️ Confusion matrix image asset could not be found.")
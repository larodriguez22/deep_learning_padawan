import streamlit as st

# ─────────────────────────── Page Config ───────────────────────────
st.set_page_config(
    page_title="Wayuunaki Translator",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Custom CSS ────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,700&family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,700;1,9..144,400&display=swap');

    /* ---------- globals ---------- */
    :root {
        --bg: #0f1117;
        --card: #1a1d27;
        --accent: #f0a500;
        --accent2: #e85d04;
        --text: #e8e6e3;
        --muted: #8b8d93;
        --border: #2a2d38;
    }

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--text);
    }

    h1, h2, h3 { font-family: 'Fraunces', serif; }

    /* ---------- header banner ---------- */
    .hero {
        background: linear-gradient(135deg, #1a1d27 0%, #2a1a0e 50%, #1a1d27 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 2.5rem 2rem 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -60%; left: -20%;
        width: 140%; height: 200%;
        background: radial-gradient(circle at 30% 40%, rgba(240,165,0,0.08) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero h1 {
        font-size: 2.4rem;
        margin: 0 0 .3rem;
        background: linear-gradient(90deg, var(--accent), var(--accent2));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p { color: var(--muted); margin: 0; font-size: 1.05rem; }

    /* ---------- translator card ---------- */
    .translator-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
    }
    .lang-label {
        font-family: 'Fraunces', serif;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: .6rem;
        display: flex;
        align-items: center;
        gap: .5rem;
    }
    .lang-label .dot {
        width: 10px; height: 10px;
        border-radius: 50%;
        display: inline-block;
    }
    .dot-spa { background: var(--accent); }
    .dot-guc { background: var(--accent2); }

    /* ---------- info cards ---------- */
    .info-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    }
    .info-card h3 {
        font-size: 1.1rem;
        margin: 0 0 .6rem;
        color: var(--accent);
    }
    .info-card p, .info-card li {
        color: var(--muted);
        font-size: .92rem;
        line-height: 1.6;
    }
    .info-card ul { padding-left: 1.2rem; }

    /* ---------- metric pills ---------- */
    .metric-row { display: flex; gap: .8rem; flex-wrap: wrap; margin: .8rem 0; }
    .metric-pill {
        background: rgba(240,165,0,0.08);
        border: 1px solid rgba(240,165,0,0.2);
        border-radius: 8px;
        padding: .45rem .9rem;
        font-size: .85rem;
        color: var(--accent);
        font-weight: 500;
    }

    /* ---------- swap button ---------- */
    .swap-btn {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 1rem 0;
    }

    /* ---------- sidebar ---------- */
    [data-testid="stSidebar"] {
        background: #13151d;
    }
    [data-testid="stSidebar"] h2 {
        color: var(--accent);
    }

    /* ---------- misc ---------- */
    .stTextArea textarea {
        background: #12141c !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 10px !important;
        font-size: 1.05rem !important;
        min-height: 140px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
        color: #000 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: .6rem 2rem !important;
        font-size: 1rem !important;
        transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(240,165,0,0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────── Sidebar ───────────────────────────────
with st.sidebar:
    st.markdown("## 🗂️ About the Project")
    st.markdown(
        """
        This translator is part of the **Deep Learning Padawan** project.
        It uses **Transfer Learning** with a fine-tuned **T5-base** model
        to translate between **Spanish** and **Wayuunaki** — the language of
        the **Wayuu** people from the Guajira Peninsula (Colombia / Venezuela).
        """
    )

    st.divider()
    st.markdown("### 📊 Training Details")
    st.markdown(
        """
        | Parameter | Value |
        |---|---|
        | Base model | `t5-base` |
        | Learning rate | `2e-5` |
        | Batch size | 2 (×8 grad accum) |
        | Optimizer | Adafactor |
        | Epochs | 10 (early stop) |
        | Metric | SacreBLEU |
        """
    )

    st.divider()
    st.markdown("### 📚 Dataset Sources")
    st.markdown(
        """
        - [Tatoeba-Challenge](https://github.com/Helsinki-NLP/Tatoeba-Challenge)
        - [Amaya, R.J.N.: Spanish-Wayuunaki](https://es.glosbe.com/guc/es/ama)
        - [OLAC / Webonary — Wayuu](https://www.webonary.org/wayuu/language/map/)
        """
    )

    st.divider()
    st.markdown("### 🔗 Links")
    st.markdown(
        """
        - [HuggingFace Dataset](https://huggingface.co/datasets/lrodriguez22/translation_spa_guc)
        - [W&B Experiments](https://wandb.ai/la-rodriguez-universidad-de-los-andes/Wayuu_spanish_translator?nw=nwuserlarodriguez)
        - [T5 Paper (JMLR)](https://jmlr.org/papers/volume21/20-074/20-074.pdf)
        """
    )

# ─────────────────────────── Hero ──────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>🌎 Wayuunaki Translator</h1>
        <p>Spanish ↔ Wayuunaki — powered by a fine-tuned T5 model &amp; Transfer Learning</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────── State ─────────────────────────────────
if "direction" not in st.session_state:
    st.session_state.direction = "spa_to_guc"  # or guc_to_spa

# ─────────────────────────── Translator UI ─────────────────────────
col_src, col_mid, col_tgt = st.columns([5, 1, 5])

with col_src:
    if st.session_state.direction == "spa_to_guc":
        src_label, src_dot = "Español (Spanish)", "dot-spa"
    else:
        src_label, src_dot = "Wayuunaki (Guc)", "dot-guc"
    st.markdown(
        f'<div class="lang-label"><span class="dot {src_dot}"></span>{src_label}</div>',
        unsafe_allow_html=True,
    )
    source_text = st.text_area(
        "source",
        placeholder="Type or paste text here…",
        label_visibility="collapsed",
        height=160,
    )

with col_mid:
    st.markdown("<div style='height:2.2rem'></div>", unsafe_allow_html=True)
    if st.button("⇄", help="Swap languages"):
        st.session_state.direction = (
            "guc_to_spa"
            if st.session_state.direction == "spa_to_guc"
            else "spa_to_guc"
        )
        st.rerun()

with col_tgt:
    if st.session_state.direction == "spa_to_guc":
        tgt_label, tgt_dot = "Wayuunaki (Guc)", "dot-guc"
    else:
        tgt_label, tgt_dot = "Español (Spanish)", "dot-spa"
    st.markdown(
        f'<div class="lang-label"><span class="dot {tgt_dot}"></span>{tgt_label}</div>',
        unsafe_allow_html=True,
    )
    output_area = st.empty()
    output_area.text_area(
        "target",
        value="",
        placeholder="Translation will appear here…",
        label_visibility="collapsed",
        height=160,
        disabled=True,
    )

# ─────────────────────────── Translate Button ──────────────────────
translate_col = st.columns([3, 2, 3])
with translate_col[1]:
    translate_clicked = st.button("🔄  Translate", use_container_width=True)

if translate_clicked and source_text.strip():
    with st.spinner("Translating…"):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            model_name = "lrodriguez22/t5-base-translation-spa-guc"

            @st.cache_resource
            def load_model_and_tokenizer(model_id):
                tok = AutoTokenizer.from_pretrained(model_id)
                mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)
                return tok, mdl

            tokenizer, model = load_model_and_tokenizer(model_name)

            if st.session_state.direction == "spa_to_guc":
                prefix = "translate spa to guc: "
            else:
                prefix = "translate guc to spa: "

            input_text = prefix + source_text
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            outputs = model.generate(**inputs, max_length=512)
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)

            output_area.text_area(
                "target",
                value=translated,
                label_visibility="collapsed",
                height=160,
                disabled=True,
            )
        except Exception as e:
            st.error(
                f"⚠️ Could not load the model. Make sure `transformers` and `torch` "
                f"are installed and you have internet access.\n\n`{e}`"
            )
elif translate_clicked:
    st.warning("Please enter some text to translate.")

# ─────────────────────────── Info Section ──────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

info1, info2 = st.columns(2)

with info1:
    st.markdown(
        """
        <div class="info-card">
            <h3>🧠 About the Model</h3>
            <p>
                This translator uses <strong>T5-base</strong>, a text-to-text
                transformer fine-tuned on a curated Spanish–Wayuunaki parallel
                corpus via Transfer Learning.
            </p>
            <div class="metric-row">
                <span class="metric-pill">T5-base (220 M params)</span>
                <span class="metric-pill">Encoder–Decoder</span>
                <span class="metric-pill">SacreBLEU eval</span>
            </div>
            <p style="margin-top:.6rem">
                Training was tracked with <strong>Weights &amp; Biases</strong>.
                See the sidebar for experiment links.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with info2:
    st.markdown(
        """
        <div class="info-card">
            <h3>🌍 The Wayuu People</h3>
            <p>
                The <strong>Wayuu</strong> are an indigenous group inhabiting
                the Guajira Peninsula in northern Colombia and northwestern
                Venezuela. <strong>Wayuunaki</strong> is their native language.
            </p>
            <p style="margin-top:.5rem">
                The parallel dataset was assembled from three open sources —
                Tatoeba-Challenge, Glosbe dictionaries, and OLAC / Webonary
                resources — and published on Hugging Face.
            </p>
            <div class="metric-row">
                <span class="metric-pill">3 data sources</span>
                <span class="metric-pill">HuggingFace Hub</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────── W&B Embed ─────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="info-card">
        <h3>📈 Training Experiments — Weights &amp; Biases</h3>
        <p style="margin-bottom:.8rem">
            Live experiment dashboard from W&amp;B showing loss curves,
            BLEU scores, and training metrics across runs.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# W&B public iframe embed
wandb_url = (
    "https://wandb.ai/la-rodriguez-universidad-de-los-andes/"
    "Wayuu_spanish_translator?nw=nwuserlarodriguez"
)
st.components.v1.iframe(wandb_url, height=620, scrolling=True)

# ─────────────────────────── Footer ────────────────────────────────
st.markdown(
    """
    <hr style="border-color: var(--border); margin: 2rem 0 1rem;">
    <p style="text-align:center; color:var(--muted); font-size:.85rem;">
        Deep Learning Padawan · Built with Streamlit · Model hosted on
        <a href="https://huggingface.co/lrodriguez22/t5-base-translation-spa-guc"
           style="color:var(--accent);" target="_blank">🤗 Hugging Face</a>
    </p>
    """,
    unsafe_allow_html=True,
)
import os, sys, warnings, traceback, html as _html, re as _re
warnings.filterwarnings("ignore")

import torch
import gradio as gr
from dotenv import load_dotenv
from groq import Groq

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

load_dotenv(os.path.join(ROOT, "secrets.env"))
HF_TOKEN   = os.getenv("HF_TOKEN")
GROQ_TOKEN = os.getenv("GROQ_API_KEY")

from pipeline.stage2a_transformer import load_transformer
from pipeline.stage2b_retriever   import load_rag_artifacts
from pipeline.router               import route_ticket, DEPT_CONF_THRESHOLD, PRIORITY_CONF_THRESHOLD
from pipeline.ui_helpers import (
    QueueWrapper, PriorityWrapper,
    render_token_heat, ig_explain, pick_top_support_words, occlusion_drop,
    build_dept_radar, build_prio_radar,
    render_routing_card, render_evidence_html, render_explanation_html,
    render_sensitivity_html, render_attribution_html, render_chat_reply,
)

device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
groq_client = Groq(api_key=GROQ_TOKEN)

model, tokenizer, queue_encoder, priority_encoder = load_transformer(
    "Nethra19/multitask-ticket-model", HF_TOKEN, device
)
retrieval_embedder, faiss_index, bm25, all_chunks, cross_encoder, priority_index, priority_chunks = load_rag_artifacts(
    "dak1299/new_rag_index", HF_TOKEN
)

queue_id_to_label    = {i: l for i, l in enumerate(queue_encoder.classes_)}
priority_id_to_label = {i: l for i, l in enumerate(priority_encoder.classes_)}
queue_ig_model    = QueueWrapper(model).to(device).eval()
priority_ig_model = PriorityWrapper(model).to(device).eval()
print(f"✓ All components ready on {device}")

TAB_KEYS   = ["evidence", "explanation", "attribution", "radar", "sensitivity"]
EMPTY_HTML = (
    "<div style='display:flex;align-items:center;justify-content:center;"
    "height:120px;color:#6B5F4B;font-size:13px;font-family:IBM Plex Sans,system-ui,sans-serif;"
    "letter-spacing:.02em;background:#15120D;border-radius:8px'>"
    "Submit a ticket to see results</div>"
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

/* ── header ── */
.app-header {
    background: linear-gradient(135deg,#15120D 0%,#241C10 50%,#15120D 100%) !important;
    border-radius: 16px !important;
    padding: 22px 30px !important;
    margin-bottom: 18px !important;
    display: flex !important;
    align-items: center !important;
    gap: 18px !important;
    border: 1px solid rgba(47,168,156,.3) !important;
    box-shadow: 0 4px 24px rgba(47,168,156,.15), 0 1px 0 rgba(255,255,255,.04) inset !important;
}
.app-header-icon {
    width: 48px !important; height: 48px !important;
    background: linear-gradient(135deg,#2FA89C,#3FC2AE) !important;
    border-radius: 12px !important;
    display: flex !important; align-items: center !important;
    justify-content: center !important; font-size: 22px !important;
    flex-shrink: 0 !important;
    box-shadow: 0 4px 14px rgba(47,168,156,.45) !important;
}
.app-header h1 {
    margin: 0 !important; font-size: 1.25rem !important;
    font-weight: 700 !important; color: #F5EEE0 !important;
    letter-spacing: -.025em !important;
    font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
}
.app-header p {
    margin: 4px 0 0 !important; font-size: 0.72rem !important;
    color: #A3947A !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: .03em !important;
}
.app-header-badge {
    margin-left: auto !important; flex-shrink: 0 !important;
    background: rgba(95,174,110,.12) !important;
    border: 1px solid rgba(95,174,110,.3) !important;
    border-radius: 99px !important; padding: 5px 14px !important;
    font-size: 11px !important; font-weight: 700 !important;
    color: #5FAE6E !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: .08em !important;
    display: flex !important; align-items: center !important; gap: 6px !important;
}
.app-header-stats {
    display: flex !important; gap: 8px !important; flex-shrink: 0 !important;
}
.stat-chip {
    display: flex !important; align-items: baseline !important; gap: 6px !important;
    background: rgba(224,163,57,.08) !important;
    border: 1px solid rgba(224,163,57,.25) !important;
    border-radius: 7px !important; padding: 5px 11px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
.stat-chip .lbl {
    font-size: 9.5px !important; color: #A3947A !important;
    text-transform: uppercase !important; letter-spacing: .08em !important;
}
.stat-chip b {
    font-size: 13px !important; color: #E0A339 !important; font-weight: 700 !important;
}

/* ── chat panel ── */
.chat-outer {
    background: #15120D !important;
    border: 1px solid #211B14 !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 20px rgba(0,0,0,.3) !important;
    margin-bottom: 0 !important;
}
.chat-head {
    background: #211B14 !important;
    padding: 11px 18px !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    border-bottom: 1px solid #3D3324 !important;
}
.chat-head-label {
    font-size: 11px !important; font-weight: 700 !important;
    color: #7A6D57 !important; letter-spacing: .1em !important;
    text-transform: uppercase !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
.chat-dot {
    width: 7px !important; height: 7px !important;
    background: #5FAE6E !important; border-radius: 50% !important;
    box-shadow: 0 0 8px rgba(95,174,110,.7) !important;
    animation: blink 2s infinite !important;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

.chat-messages {
    height: 320px !important;
    overflow-y: auto !important;
    padding: 16px 18px !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 12px !important;
    scroll-behavior: smooth !important;
}
.chat-messages::-webkit-scrollbar { width: 3px; }
.chat-messages::-webkit-scrollbar-track { background: transparent; }
.chat-messages::-webkit-scrollbar-thumb { background: #3D3324; border-radius: 2px; }

.msg-user {
    align-self: flex-end !important;
    max-width: 80% !important;
    background: linear-gradient(135deg,#1F7E74,#2FA89C) !important;
    color: #F5EEE0 !important;
    border-radius: 14px 14px 4px 14px !important;
    padding: 10px 14px !important;
    font-size: 13px !important; line-height: 1.55 !important;
    box-shadow: 0 3px 10px rgba(31,126,116,.4) !important;
    font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
}
.msg-bot {
    align-self: flex-start !important;
    max-width: 88% !important;
    background: #211B14 !important;
    color: #CBBFA5 !important;
    border: 1px solid #3D3324 !important;
    border-radius: 4px 14px 14px 14px !important;
    padding: 10px 14px !important;
    font-size: 13px !important; line-height: 1.65 !important;
    font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
}
.msg-bot strong { color: #EFE7D8 !important; }
.msg-bot code {
    background: #15120D !important; color: #7FD4C4 !important;
    padding: 1px 6px !important; border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important; font-size: 11px !important;
}
.chat-empty-state {
    flex: 1 !important;
    display: flex !important; flex-direction: column !important;
    align-items: center !important; justify-content: center !important;
    gap: 10px !important; color: #3D3324 !important;
    font-size: 13px !important; padding: 20px 0 !important;
    font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
}
.chat-empty-state .icon { font-size: 36px !important; opacity: .35 !important; }
.chat-empty-state .hint { color: #6B5F4B !important; font-size: 12px !important; margin-top: 2px !important; }

/* ── example buttons ── */
.example-btn {
    width: 100% !important;
    text-align: left !important;
    justify-content: flex-start !important;
    border-radius: 8px !important;
    background: #211B14 !important;
    color: #CBBFA5 !important;
    border: 1px solid #3D3324 !important;
    font-size: 12px !important;
    padding: 8px 12px !important;
    font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
    transition: background .15s, border-color .15s !important;
    margin-bottom: 4px !important;
    cursor: pointer !important;
    display: block !important;
}
.example-btn:hover {
    background: #2E2619 !important;
    border-color: #2FA89C !important;
    color: #F5EEE0 !important;
}

/* ── tab strip ── */
#tab-strip {
    display: flex !important; gap: 4px !important;
    margin-bottom: 10px !important;
    flex-wrap: wrap !important;
    background: #211B14 !important;
    border-radius: 10px !important;
    padding: 6px !important;
    border: 1px solid #3D3324 !important;
}
#tab-strip button {
    border-radius: 7px !important;
    padding: 7px 13px !important;
    font-size: 0.79rem !important;
    font-weight: 600 !important;
    letter-spacing: .01em !important;
    white-space: nowrap !important;
    font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
    background: transparent !important;
    color: #A3947A !important;
    border: 1px solid transparent !important;
    transition: all .15s ease !important;
}
#tab-strip button:hover {
    background: #2E2619 !important;
    color: #EFE7D8 !important;
}
#tab-strip button.primary {
    background: linear-gradient(135deg,#1F7E74,#2FA89C) !important;
    color: #fff !important;
    border-color: rgba(47,168,156,.4) !important;
    box-shadow: 0 2px 8px rgba(47,168,156,.35) !important;
}
"""

EXAMPLES = [
    "I was charged twice for my subscription last month. Please refund the duplicate payment.",
    "My internet isn't working and I can't log in — urgent deadline today.",
    "We need to update payroll records for three new hires this week.",
    "The checkout page keeps crashing when customers try to pay with Visa.",
    "Can you tell me your standard service hours and how to escalate?",
    "Our Salesforce CRM has been down since 9am, sales team is blocked.",
    "URGENT: Our entire payment processing system has been down for 3 hours. We are losing thousands of dollars per minute and cannot process any customer transactions. This is a critical outage affecting all 500 of our enterprise clients. We need immediate escalation to your senior technical team. Our SLA breach deadline is in 2 hours."
]


def _md_to_html(text):
    safe = _html.escape(str(text))
    safe = safe.replace("\n\n", "<br><br>").replace("\n", "<br>")
    safe = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
    safe = _re.sub(r"`(.+?)`", r"<code>\1</code>", safe)
    return safe


def _render_messages(history):
    if not history:
        return (
            "<div class='chat-empty-state'>"
            "<div class='icon'>🎫</div>"
            "<div>Submit a support ticket to begin</div>"
            "<div class='hint'>Your conversation will appear here</div>"
            "</div>"
        )
    parts = []
    for turn in history:
        role    = turn.get("role", "")
        content = _md_to_html(turn.get("content", ""))
        css_cls = "msg-user" if role == "user" else "msg-bot"
        parts.append(f"<div class='{css_cls}'>{content}</div>")
    scroll = ("<script>setTimeout(()=>{"
              "var el=document.querySelector('.chat-messages');"
              "if(el)el.scrollTop=el.scrollHeight;},60)</script>")
    return "".join(parts) + scroll


def _build_chat_panel(history):
    inner = _render_messages(history)
    return f"""
<div class="chat-outer">
  <div class="chat-head">
    <div class="chat-dot"></div>
    <span class="chat-head-label">Ticket Conversation</span>
  </div>
  <div class="chat-messages">{inner}</div>
</div>"""


_EMPTY_CHAT = _build_chat_panel([])


THEME = gr.themes.Base().set(
    body_background_fill="#15120D", body_background_fill_dark="#15120D",
    body_text_color="#F5EEE0", body_text_color_dark="#F5EEE0",
    body_text_color_subdued="#A3947A", body_text_color_subdued_dark="#A3947A",
    background_fill_primary="#211B14", background_fill_primary_dark="#211B14",
    background_fill_secondary="#2E2619", background_fill_secondary_dark="#2E2619",
    block_background_fill="#211B14", block_background_fill_dark="#211B14",
    block_border_color="#3D3324", block_border_color_dark="#3D3324",
    block_label_text_color="#A3947A", block_label_text_color_dark="#A3947A",
    block_title_text_color="#A3947A", block_title_text_color_dark="#A3947A",
    border_color_primary="#3D3324", border_color_primary_dark="#3D3324",
    panel_background_fill="#211B14", panel_background_fill_dark="#211B14",
    panel_border_color="#3D3324", panel_border_color_dark="#3D3324",
    accordion_text_color="#F5EEE0", accordion_text_color_dark="#F5EEE0",
    input_background_fill="#15120D", input_background_fill_dark="#15120D",
    input_border_color="#3D3324", input_border_color_dark="#3D3324",
    input_border_color_focus="#2FA89C", input_border_color_focus_dark="#2FA89C",
    input_placeholder_color="#7A6D57", input_placeholder_color_dark="#7A6D57",
    button_primary_background_fill="#2FA89C", button_primary_background_fill_dark="#2FA89C",
    button_primary_background_fill_hover="#3FC2AE", button_primary_background_fill_hover_dark="#3FC2AE",
    button_primary_text_color="#15120D", button_primary_text_color_dark="#15120D",
    button_secondary_background_fill="#2E2619", button_secondary_background_fill_dark="#2E2619",
    button_secondary_background_fill_hover="#3D3324", button_secondary_background_fill_hover_dark="#3D3324",
    button_secondary_text_color="#F5EEE0", button_secondary_text_color_dark="#F5EEE0",
    button_secondary_border_color="#3D3324", button_secondary_border_color_dark="#3D3324",
    link_text_color="#2FA89C", link_text_color_dark="#2FA89C",
)

with gr.Blocks(theme=THEME, css=CSS, title="Support Ticket Router") as demo:

    chat_history = gr.State([])

    gr.HTML(f"""
    <div class="app-header">
      <div class="app-header-icon">🎫</div>
      <div>
        <h1>Customer Support Ticket Router</h1>
        <p>DistilBERT · HNSW + BM25 + CrossEncoder RRF · Groq LLaMA-3.3-70b</p>
      </div>
      <div class="app-header-stats">
        <div class="stat-chip"><span class="lbl">Dept gate</span><b>{DEPT_CONF_THRESHOLD:.2f}</b></div>
        <div class="stat-chip"><span class="lbl">Priority gate</span><b>{PRIORITY_CONF_THRESHOLD:.2f}</b></div>
      </div>
      <div class="app-header-badge"><span style="width:7px;height:7px;background:#5FAE6E;border-radius:50%;display:inline-block"></span> LIVE</div>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── LEFT column ──────────────────────────────────────────────────────
        with gr.Column(scale=4, min_width=320):
            chat_display = gr.HTML(value=_EMPTY_CHAT)
            msg_input = gr.Textbox(
                placeholder="Describe your issue…",
                lines=3, max_lines=6, show_label=False,
            )
            with gr.Row():
                submit_btn = gr.Button("⚡  Submit Ticket", variant="primary", scale=3)
                clear_btn  = gr.Button("Clear", variant="secondary", scale=1)
            with gr.Accordion("Example tickets", open=False):
                example_btns = []
                for ex in EXAMPLES:
                    btn = gr.Button(
                        ex[:72] + ("…" if len(ex) > 72 else ""),
                        variant="secondary",
                        elem_classes=["example-btn"],
                    )
                    example_btns.append((btn, ex))

        # ── RIGHT column ─────────────────────────────────────────────────────
        with gr.Column(scale=6, min_width=420):

            routing_card = gr.HTML(value=EMPTY_HTML)
            active_panel = gr.State("evidence")

            with gr.Row(elem_id="tab-strip"):
                btn_evidence    = gr.Button("📊 Evidence",    variant="primary")
                btn_explanation = gr.Button("🔍 Explanation", variant="secondary")
                btn_attribution = gr.Button("🧠 Attribution", variant="secondary")
                btn_radar       = gr.Button("📡 Radar",       variant="secondary")
                btn_sensitivity = gr.Button("⚖️ Sensitivity", variant="secondary")

            panel_evidence    = gr.Column(visible=True)
            panel_explanation = gr.Column(visible=False)
            panel_attribution = gr.Column(visible=False)
            panel_radar       = gr.Column(visible=False)
            panel_sensitivity = gr.Column(visible=False)

            with panel_evidence:
                evidence_html = gr.HTML(value=EMPTY_HTML)
            with panel_explanation:
                explanation_html = gr.HTML(value=EMPTY_HTML)
            with panel_attribution:
                attribution_html = gr.HTML(value=EMPTY_HTML)
            with panel_radar:
                radar_dept = gr.Plot(value=None)
                radar_prio = gr.Plot(value=None)
            with panel_sensitivity:
                sensitivity_html = gr.HTML(value=EMPTY_HTML)

            tab_outputs = [
                active_panel,
                btn_evidence, btn_explanation, btn_attribution, btn_radar, btn_sensitivity,
                panel_evidence, panel_explanation, panel_attribution, panel_radar, panel_sensitivity,
            ]
            btn_evidence.click(   fn=lambda: _set_tab("evidence"),    inputs=[], outputs=tab_outputs)
            btn_explanation.click(fn=lambda: _set_tab("explanation"), inputs=[], outputs=tab_outputs)
            btn_attribution.click(fn=lambda: _set_tab("attribution"), inputs=[], outputs=tab_outputs)
            btn_radar.click(      fn=lambda: _set_tab("radar"),       inputs=[], outputs=tab_outputs)
            btn_sensitivity.click(fn=lambda: _set_tab("sensitivity"), inputs=[], outputs=tab_outputs)

    def _set_tab(selected):
        b = lambda k: gr.update(variant="primary" if selected == k else "secondary")
        v = lambda k: gr.update(visible=(selected == k))
        return (
            selected,
            b("evidence"), b("explanation"), b("attribution"), b("radar"), b("sensitivity"),
            v("evidence"), v("explanation"), v("attribution"), v("radar"), v("sensitivity"),
        )

    def process_ticket_ui(user_message, history):
        history = list(history or [])
        if not user_message or not user_message.strip():
            return (history, _build_chat_panel(history),
                    EMPTY_HTML, EMPTY_HTML, EMPTY_HTML, EMPTY_HTML, None, None, EMPTY_HTML)

        history.append({"role": "user", "content": user_message.strip()})

        try:
            result = route_ticket(
                user_message.strip(), groq_client,
                model, tokenizer, queue_encoder, priority_encoder, device,
                retrieval_embedder,
                faiss_index, bm25, all_chunks, cross_encoder,
                priority_index, priority_chunks
            )
        except Exception as e:
            traceback.print_exc()
            history.append({"role": "assistant", "content": f"⚠️ Pipeline error: {e}"})
            err = f"<div style='color:#E2645A;padding:16px;font-size:13px'>⚠️ {e}</div>"
            return (history, _build_chat_panel(history), err, err, err, err, None, None, err)

        history.append({"role": "assistant", "content": render_chat_reply(result)})

        # Attribution
        q_idx = next((i for i, l in queue_id_to_label.items() if l == result.transformer_dept), None)
        support_words, queue_html = [], ""
        if q_idx is not None:
            q_attr, q_err = ig_explain(result.cleaned_text, queue_ig_model, q_idx, tokenizer, device)
            queue_html = (render_token_heat(q_attr["ordered"],
                f"Department attribution → {result.transformer_dept}",
                "Blue = supports · Red = pushes against (Integrated Gradients)")
                if q_attr else render_token_heat([], "IG unavailable", subtitle=q_err))
            if q_attr:
                support_words = pick_top_support_words(q_attr["sorted"], k=3)

        p_idx = next((i for i, l in priority_id_to_label.items() if l == result.priority), None)
        prio_html = ""
        if p_idx is not None:
            p_attr, p_err = ig_explain(result.cleaned_text, priority_ig_model, p_idx, tokenizer, device)
            prio_html = (render_token_heat(p_attr["ordered"],
                f"Priority attribution → {result.priority}", "Same colour logic as above.")
                if p_attr else render_token_heat([], "IG unavailable", subtitle=p_err))

        occl_rows = occlusion_drop(
            result.cleaned_text, support_words,
            result.transformer_conf, 0.5,
            model, tokenizer, queue_encoder, priority_encoder, device
        )

        return (
            history,
            _build_chat_panel(history),
            render_routing_card(result),
            render_evidence_html(result),
            render_explanation_html(result),
            render_attribution_html(queue_html, prio_html, occl_rows),
            build_dept_radar(result),
            build_prio_radar(result),
            render_sensitivity_html(result, support_words),
        )

    def clear_ui():
        return ([], _EMPTY_CHAT, EMPTY_HTML, EMPTY_HTML, EMPTY_HTML, EMPTY_HTML, None, None, EMPTY_HTML)

    OUTS = [
        chat_history,
        chat_display,
        routing_card,
        evidence_html,
        explanation_html,
        attribution_html,
        radar_dept,
        radar_prio,
        sensitivity_html,
    ]

    submit_btn.click(fn=process_ticket_ui, inputs=[msg_input, chat_history], outputs=OUTS
        ).then(fn=lambda: "", inputs=[], outputs=msg_input)
    msg_input.submit(fn=process_ticket_ui, inputs=[msg_input, chat_history], outputs=OUTS
        ).then(fn=lambda: "", inputs=[], outputs=msg_input)
    clear_btn.click(fn=clear_ui, inputs=[], outputs=OUTS
        ).then(fn=lambda: "", inputs=[], outputs=msg_input)

    for _ebtn, _etxt in example_btns:
        _ebtn.click(fn=lambda t=_etxt: t, inputs=[], outputs=msg_input)

if __name__ == "__main__":
    demo.launch(server_port=7860, show_error=True, share=False)
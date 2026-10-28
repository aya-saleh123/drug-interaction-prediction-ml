"""
Floating Chatbot Widget with n8n Integration & Local ML Analysis
Embedded seamlessly inside the Streamlit Web Application.
"""

from __future__ import annotations
import json
import re
import uuid
import requests
import streamlit as st
import streamlit.components.v1 as components

# Arabic drug transliterations mapping to generic English names
ARABIC_DRUG_MAP = {
    "اسبرين": "Aspirin",
    "أسبرين": "Aspirin",
    "وارفارين": "Warfarin",
    "باراسيتامول": "Acetaminophen",
    "بانادول": "Acetaminophen",
    "بنادول": "Acetaminophen",
    "بروفين": "Ibuprofen",
    "ايبوبروفين": "Ibuprofen",
    "ميتفورمين": "Metformin",
    "اوميبرازول": "Omeprazole",
    "أوميبرازول": "Omeprazole",
    "اتورفاستاتين": "Atorvastatin",
    "أملوديبين": "Amlodipine",
    "اميلوديبين": "Amlodipine",
    "كلوبيدوجريل": "Clopidogrel",
    "بلافيكس": "Clopidogrel",
    "ليسينوبريل": "Lisinopril",
    "سيمفاستاتين": "Simvastatin",
    "ديازيبام": "Diazepam",
    "فالبورات": "Valproic acid",
    "سيبروفلوكساسين": "Ciprofloxacin",
    "ازيثرومايسين": "Azithromycin",
    "اموكسيسيلين": "Amoxicillin",
    "ترامادول": "Tramadol",
    "ديجوكسين": "Digoxin",
    "فلوكسيتين": "Fluoxetine",
    "انسولين": "Insulin",
}

STOP_WORDS = {
    "can", "take", "with", "and", "the", "between", "interaction", "what",
    "is", "does", "have", "effects", "side", "patient", "using", "check",
    "هل", "ينفع", "اخذ", "أخذ", "مع", "و", "بين", "تعارض", "تفاعل", "ما",
    "هو", "اثر", "اعراض", "اثار", "جانبية", "ممكن", "دواء", "علاج", "حبوب"
}


def _extract_drugs_from_text(text: str, drug_lookup: dict[str, str]) -> list[str]:
    """Identify candidate drug names from user text."""
    found: list[str] = []
    text_lower = text.lower()

    # 1. Match Arabic names
    for ar_term, en_name in ARABIC_DRUG_MAP.items():
        if ar_term in text_lower and en_name not in found:
            found.append(en_name)

    # 2. Match English words against known drug names
    words = re.findall(r"[a-zA-Z]{3,}", text)
    known_names_lower = {name.lower(): name for name in drug_lookup.values()}

    for word in words:
        w_lower = word.lower()
        if w_lower in STOP_WORDS:
            continue
        # Direct lookup
        if w_lower in known_names_lower:
            matched_real_name = known_names_lower[w_lower]
            if matched_real_name not in found:
                found.append(matched_real_name)
        elif len(word) >= 4 and word.capitalize() not in found:
            # Add capitalized token as candidate if no better match
            found.append(word.capitalize())

    return found


def _query_local_model(api_url: str, drug_a: str, drug_b: str) -> str:
    """Query the FastAPI backend for drug interaction."""
    try:
        endpoint = f"{api_url.rstrip('/')}/predict-by-name"
        res = requests.post(
            endpoint,
            json={"drug_a": drug_a, "drug_b": drug_b},
            timeout=30,
        )
        if res.status_code == 200:
            data = res.json()
            resolved = data.get("resolved_inputs", {})
            name_a = resolved.get("drug_a", {}).get("matched_name", drug_a)
            name_b = resolved.get("drug_b", {}).get("matched_name", drug_b)
            severity = data.get("severity", "Unknown")

            # Severity badge styling
            severity_badge = {
                "Major": "🔴 **عالي الخطورة (Major)** - قد يسبب مضاعفات خطيرة تتطلب عناية طبية وتعديل الجرعة.",
                "Moderate": "🟠 **متوسط الخطورة (Moderate)** - يتطلب مراقبة سريرية أو تجنب التزامن قدر الإمكان.",
                "Minor": "🟢 **منخفض الخطورة (Minor)** - تفاعل طفيف ذو تأثير سريري محدود.",
            }.get(severity, f"⚪ **{severity}**")

            # Top symptoms
            ranked = data.get("ranked_adverse_effects", [])[:5]
            symptoms_text = ""
            if ranked:
                symptoms_text = "\n\n**🩺 أبرز الأعراض الجانبية المحتملة:**\n"
                for item in ranked:
                    name = item.get("name", "Unknown")
                    prob = item.get("probability", 0) * 100
                    symptoms_text += f"- **{name}** (احتمال: {prob:.1f}%)\n"

            return (
                f"### 💊 نتائج التفاعل بين `{name_a}` و `{name_b}`\n\n"
                f"**درجة الخطورة:**\n{severity_badge}"
                f"{symptoms_text}\n\n"
                f"> ⚠️ **تنبيه:** هذا التنبؤ ناتج عن نموذج ذكاء اصطناعي بحثي ولا يُعد بديلاً عن استشارة الطبيب أو الصيدلي."
            )
        else:
            detail = res.json().get("detail", res.text)
            return f"⚠️ تعذر العثور على الدواء أو حساب التفاعل:\n`{detail}`\n\nيرجى التأكد من كتابة اسم الدواء العلمي بالإنجليزية أو الاختيار من القائمة."
    except Exception as exc:
        return f"❌ خطأ في الاتصال بالسيرفر المحلي: {exc}"


def _query_n8n_webhook(webhook_url: str, user_text: str, session_id: str) -> str:
    """Send user message to external n8n Chatbot Workflow."""
    try:
        payload = {
            "action": "sendMessage",
            "chatInput": user_text,
            "message": user_text,
            "sessionId": session_id,
        }
        res = requests.post(webhook_url.strip(), json=payload, timeout=45)
        if res.status_code == 200:
            try:
                data = res.json()
                if isinstance(data, dict):
                    return (
                        data.get("output")
                        or data.get("text")
                        or data.get("message")
                        or data.get("response")
                        or json.dumps(data, ensure_ascii=False, indent=2)
                    )
                elif isinstance(data, str):
                    return data
                elif isinstance(data, list) and data:
                    first = data[0]
                    if isinstance(first, dict):
                        return first.get("output") or first.get("text") or str(first)
                    return str(first)
                return str(data)
            except ValueError:
                return res.text
        elif res.status_code == 404:
            return (
                "⚠️ **تنبيه من n8n:** الـ Webhook غير مسجل حالياً (404).\n\n"
                "📌 **الحل:**\n"
                "1. تأكد من تفعيل الـ Workflow بالضغط على زر **Publish / Active** (الزر الأخضر في أعلى يمين شاشة n8n).\n"
                "2. إذا كنت تختبر السير ولم تفعله بعد، استخدم رابط **Test URL** أو اضغط على **Test this trigger** في n8n."
            )
        else:
            return f"⚠️ استجاب سيرفر n8n برمز ({res.status_code}):\n`{res.text}`"
    except Exception as exc:
        return f"❌ تعذر الاتصال بـ n8n Webhook: {exc}"


def render_floating_chatbot(api_url: str, drugs: list[dict[str, str]]):
    """Render the floating chatbot widget at the bottom right of the page."""

    # Initialize Session States
    if "chatbot_messages" not in st.session_state:
        st.session_state.chatbot_messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 مرحباً بك! أنا **المساعد الذكي للتفاعلات الدوائية** 💊\n\n"
                    "يمكنك سؤالي عن أي تفاعل بين دواءين (مثل: *هل يتعارض الأسبرين مع الوارفارين؟* أو *Aspirin and Warfarin*).\n\n"
                    "إذا قمت بربط **n8n**، سأقوم بالتواصل معه تلقائياً!"
                ),
            }
        ]

    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = str(uuid.uuid4())

    if "n8n_webhook_url" not in st.session_state:
        st.session_state.n8n_webhook_url = "https://motamer540.app.n8n.cloud/webhook/d0a17e2f-9886-4ecf-ac56-b3c5f5010218/chat"

    if "use_n8n_mode" not in st.session_state:
        st.session_state.use_n8n_mode = True

    if "embed_n8n_official" not in st.session_state:
        st.session_state.embed_n8n_official = False

    # Lookup mapping for quick drug matching
    drug_lookup = {
        str(d.get("drug_id")): str(d.get("drug_name") or d.get("label") or "")
        for d in drugs
    }

    # Inject FAB (Floating Action Button) and Popover CSS
    st.markdown(
        """
        <style>
        /* Floating Chatbot Popover Styling */
        div[data-testid="stPopover"] {
            position: fixed !important;
            bottom: 25px !important;
            right: 25px !important;
            z-index: 999999 !important;
        }

        div[data-testid="stPopover"] > button {
            background: linear-gradient(135deg, #10b981 0%, #047857 100%) !important;
            color: #ffffff !important;
            border: 2px solid rgba(255, 255, 255, 0.4) !important;
            border-radius: 50px !important;
            padding: 14px 26px !important;
            font-size: 16px !important;
            font-weight: 700 !important;
            letter-spacing: 0.3px !important;
            box-shadow: 0 10px 30px rgba(16, 185, 129, 0.45) !important;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
            display: flex !important;
            align-items: center !important;
            gap: 10px !important;
        }

        div[data-testid="stPopover"] > button:hover {
            transform: translateY(-4px) scale(1.04) !important;
            box-shadow: 0 16px 36px rgba(16, 185, 129, 0.6) !important;
            background: linear-gradient(135deg, #059669 0%, #065f46 100%) !important;
        }

        div[data-testid="stPopoverBody"] {
            width: 440px !important;
            max-width: 90vw !important;
            max-height: 640px !important;
            border-radius: 18px !important;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.25) !important;
            border: 1px solid rgba(226, 232, 240, 0.8) !important;
            overflow-y: auto !important;
            padding: 16px !important;
        }

        .chat-bubble-user {
            background-color: #2563eb;
            color: white;
            border-radius: 14px 14px 2px 14px;
            padding: 10px 14px;
            margin: 6px 0;
            display: inline-block;
            max-width: 85%;
            font-size: 14px;
            float: right;
            clear: both;
        }

        .chat-bubble-bot {
            background-color: #f1f5f9;
            color: #0f172a;
            border-radius: 14px 14px 14px 2px;
            padding: 12px 16px;
            margin: 6px 0;
            display: inline-block;
            max-width: 90%;
            font-size: 14px;
            border: 1px solid #e2e8f0;
            float: left;
            clear: both;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Optional: Official n8n Chat Widget embed if enabled
    if (
        st.session_state.embed_n8n_official
        and st.session_state.n8n_webhook_url.strip()
    ):
        n8n_url = st.session_state.n8n_webhook_url.strip()
        components.html(
            f"""
            <link href="https://cdn.jsdelivr.net/npm/@n8n/chat/dist/style.css" rel="stylesheet" />
            <script type="module">
                import {{ createChat }} from 'https://cdn.jsdelivr.net/npm/@n8n/chat/dist/chat.bundle.es.js';
                createChat({{
                    webhookUrl: '{n8n_url}',
                    mode: 'window',
                    showWelcomeScreen: true,
                    defaultLanguage: 'ar',
                    initialMessages: [
                        'مرحباً بك! أنا مساعد التفاعلات الدوائية عبر n8n 🤖',
                        'اسألني عن أي دواء أو تعارض دوائي.'
                    ],
                    i18n: {{
                        ar: {{
                            title: 'مساعد الأدوية الذكي',
                            subtitle: 'متصل بنظام n8n للذكاء الاصطناعي',
                            inputPlaceholder: 'اكتب سؤالك هنا...',
                        }}
                    }}
                }});
            </script>
            """,
            height=0,
        )

    # Floating Popover Button (Bottom-Right)
    badge_label = "💬 الشات بوت الطبي (Chatbot)"
    with st.popover(badge_label, use_container_width=False):
        st.markdown("### 🤖 المساعد الذكي للتفاعلات الدوائية")
        st.caption("يعمل بالذكاء الاصطناعي ويدعم الربط المباشر مع **n8n**")

        # Quick Settings Accordion
        with st.expander("⚙️ إعدادات الربط مع n8n (n8n Integration)", expanded=False):
            st.session_state.use_n8n_mode = st.toggle(
                "تفعيل الردود عبر n8n (Use n8n)",
                value=st.session_state.use_n8n_mode,
                help="عند التفعيل، سيتم إرسال رسائل المحادثة إلى رابط الـ Webhook الخاص بك في n8n.",
            )

            n8n_input = st.text_input(
                "رابط n8n Webhook URL:",
                value=st.session_state.n8n_webhook_url,
                placeholder="https://your-n8n.com/webhook/xxxx",
                help="الرابط الذي يتم توليده في عقدة Webhook أو Chat Trigger في n8n.",
            )
            if n8n_input != st.session_state.n8n_webhook_url:
                st.session_state.n8n_webhook_url = n8n_input

            st.session_state.embed_n8n_official = st.checkbox(
                "تضمين ويدجت n8n الرسمي (Official @n8n/chat Widget)",
                value=st.session_state.embed_n8n_official,
                help="يُظهر زر الـ Chat الخاص بـ n8n أسفل الشاشة مباشرة.",
            )

            if st.button("🗑️ مسح المحادثة"):
                st.session_state.chatbot_messages = [
                    {
                        "role": "assistant",
                        "content": "تم مسح المحادثة. كيف يمكنني مساعدتك؟",
                    }
                ]
                st.rerun()

        # Display Message History
        chat_container = st.container(height=320)
        with chat_container:
            for msg in st.session_state.chatbot_messages:
                avatar = "🧑‍💻" if msg["role"] == "user" else "💊"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])

        # Quick action prompts
        st.markdown("**أمثلة سريعة للتجربة:**")
        chip_col1, chip_col2 = st.columns(2)
        with chip_col1:
            if st.button("💊 Aspirin + Warfarin", use_container_width=True):
                _handle_user_message(
                    "هل يتعارض Aspirin مع Warfarin؟",
                    api_url,
                    drug_lookup,
                )
                st.rerun()
        with chip_col2:
            if st.button("💊 Metformin + Glipizide", use_container_width=True):
                _handle_user_message(
                    "فحص تفاعل Metformin مع Glipizide",
                    api_url,
                    drug_lookup,
                )
                st.rerun()

        # Message Input Form
        with st.form("floating_chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "رسالتك:",
                placeholder="اكتب أسماء الأدوية أو سؤالك الطبي...",
                label_visibility="collapsed",
            )
            submit_btn = st.form_submit_button("إرسال 🚀", use_container_width=True)

            if submit_btn and user_input.strip():
                _handle_user_message(user_input.strip(), api_url, drug_lookup)
                st.rerun()


def _handle_user_message(
    user_text: str, api_url: str, drug_lookup: dict[str, str]
):
    """Process user input either via n8n or local ML model."""
    # 1. Add user message
    st.session_state.chatbot_messages.append(
        {"role": "user", "content": user_text}
    )

    # 2. Check if n8n mode is enabled
    if (
        st.session_state.use_n8n_mode
        and st.session_state.n8n_webhook_url.strip()
    ):
        bot_response = _query_n8n_webhook(
            st.session_state.n8n_webhook_url.strip(),
            user_text,
            st.session_state.chat_session_id,
        )
    else:
        # Local ML extraction & prediction
        drugs_found = _extract_drugs_from_text(user_text, drug_lookup)

        if len(drugs_found) >= 2:
            drug_a = drugs_found[0]
            drug_b = drugs_found[1]
            bot_response = _query_local_model(api_url, drug_a, drug_b)
        elif len(drugs_found) == 1:
            drug = drugs_found[0]
            bot_response = (
                f"وجدت دواءً واحداً في سؤالك: **{drug}**.\n\n"
                f"لفحص التفاعل الدوائي، يرجى كتابة **اسم الدواء الثاني** معه (مثلاً: *{drug} مع Aspirin*)."
            )
        else:
            bot_response = (
                "لم أتعرف على أسماء الأدوية بدقة في رسالتك.\n\n"
                "💡 **كيف تسألني؟**\n"
                "- اذكر اسمين لأي دواءين بالإنجليزية أو العربية (مثل: *Aspirin and Warfarin* أو *اسبرين مع باراسيتامول*).\n"
                "- أو قم بربط **n8n** من الإعدادات بأعلى النافذة لتحصل على ردود ذكاء اصطناعي تفاعلية متقدمة!"
            )

    # 3. Add bot response
    st.session_state.chatbot_messages.append(
        {"role": "assistant", "content": bot_response}
    )

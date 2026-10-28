
import os
import pandas as pd
import requests
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

DRUGS_URL = f"{API_URL}/drugs"
PREDICT_URL = f"{API_URL}/predict"

st.set_page_config(
    page_title="Drug Interaction Predictor",
    page_icon="💊",
    layout="wide",
)


# --------------------------------------------------
# Page header
# --------------------------------------------------

st.title("💊 Drug Interaction Predictor")

st.write(
    "Select two medicines to explore the model's predicted "
    "interaction severity and potential adverse effects."
)

st.warning(
    "Research prototype only. Predictions are not medical advice "
    "and should not be used to make treatment decisions. "
    "Consult a qualified healthcare professional about drug interactions."
)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def get_drugs():
    """Retrieve the medicine catalog from the FastAPI backend."""
    try:
        response = requests.get(
            DRUGS_URL,
            params={"search": "", "limit": 600},
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()

        # Support either a direct list or a wrapped list response.
        if isinstance(data, dict):
            data = data.get("drugs", [])

        if not isinstance(data, list):
            st.error("The backend returned an unexpected drug-list format.")
            return []

        # Keep only entries with the fields the frontend needs.
        return [
            drug for drug in data
            if isinstance(drug, dict)
            and drug.get("drug_id") is not None
        ]

    except requests.RequestException as exc:
        st.error(
            "Could not connect to the prediction API. "
            "Make sure the FastAPI backend is running."
        )
        st.caption(f"Connection details: {exc}")
        return []


def display_drug_name(drug_id, drug_by_id):
    """Display the actual medicine name, never the description."""
    drug = drug_by_id.get(drug_id, {})
    return drug.get("drug_name") or drug.get("label") or drug_id


def get_drug_description(drug_id, drug_by_id):
    """Retrieve the description separately from the medicine name."""
    drug = drug_by_id.get(drug_id, {})
    return (
        drug.get("description")
        or "No description is available for this medicine."
    )


def filter_drugs(search_text, drug_ids, drug_by_id):
    """Filter medicine options using one search field's text."""
    search_text = search_text.strip().casefold()

    if not search_text:
        return drug_ids

    matches = []

    for drug_id in drug_ids:
        drug = drug_by_id[drug_id]

        name = str(
            drug.get("drug_name") or drug.get("label") or ""
        )
        description = str(drug.get("description") or "")
        profile_id = str(drug_id)

        searchable_text = (
            f"{name} {description} {profile_id}"
        ).casefold()

        if search_text in searchable_text:
            matches.append(drug_id)

    return matches


# --------------------------------------------------
# Load medicine catalog
# --------------------------------------------------

drugs = get_drugs()

if not drugs:
    st.info(
        "No medicines are available yet. Start the backend and "
        "refresh this page."
    )
    st.stop()

drug_by_id = {
    str(drug["drug_id"]): drug
    for drug in drugs
}

drug_ids = list(drug_by_id.keys())

# Sort by actual medicine name for easier browsing.
drug_ids.sort(
    key=lambda drug_id: display_drug_name(
        drug_id, drug_by_id
    ).casefold()
)


# --------------------------------------------------
# Drug selection
# Each medicine has its own independent search input.
# --------------------------------------------------

st.subheader("Select medicines")

col_a, col_b = st.columns(2, gap="large")


# --------------------------------------------------
# Drug A
# --------------------------------------------------

with col_a:
    st.markdown("### 🅰️ Drug A")

    search_a = st.text_input(
        "Search for Drug A",
        key="search_drug_a",
        placeholder="Type a medicine name or profile ID...",
    )

    options_a = filter_drugs(
        search_a,
        drug_ids,
        drug_by_id,
    )

    if options_a:
        drug_a_id = st.selectbox(
            "Choose Drug A",
            options=options_a,
            format_func=lambda drug_id: display_drug_name(
                drug_id, drug_by_id
            ),
            key="select_drug_a",
        )

        st.caption("Drug A description")

        st.write(
            get_drug_description(drug_a_id, drug_by_id)
        )

    else:
        drug_a_id = None
        st.info("No medicines match the Drug A search.")


# --------------------------------------------------
# Drug B
# --------------------------------------------------

with col_b:
    st.markdown("### 🅱️ Drug B")

    search_b = st.text_input(
        "Search for Drug B",
        key="search_drug_b",
        placeholder="Type a medicine name or profile ID...",
    )

    options_b = filter_drugs(
        search_b,
        drug_ids,
        drug_by_id,
    )

    if options_b:
        drug_b_id = st.selectbox(
            "Choose Drug B",
            options=options_b,
            format_func=lambda drug_id: display_drug_name(
                drug_id, drug_by_id
            ),
            key="select_drug_b",
        )

        st.caption("Drug B description")

        st.write(
            get_drug_description(drug_b_id, drug_by_id)
        )

    else:
        drug_b_id = None
        st.info("No medicines match the Drug B search.")


# --------------------------------------------------
# Prediction
# Keep the existing API request format unchanged.
# --------------------------------------------------

st.divider()

if st.button(
    "🔍 Predict Drug Interaction",
    type="primary",
    use_container_width=True,
):
    if not drug_a_id or not drug_b_id:
        st.error(
            "Please select a medicine for both Drug A and Drug B."
        )

    elif drug_a_id == drug_b_id:
        st.warning(
            "Please select two different medicines to predict "
            "their interaction."
        )

    else:
        with st.spinner("Analyzing the selected medicines..."):
            try:
                response = requests.post(
                    PREDICT_URL,
                    json={
                        "drug_a_id": drug_a_id,
                        "drug_b_id": drug_b_id,
                    },
                    timeout=120,
                )

                response.raise_for_status()
                result = response.json()

                st.session_state["interaction_result"] = result
                st.session_state["interaction_drug_a"] = drug_a_id
                st.session_state["interaction_drug_b"] = drug_b_id

            except requests.HTTPError as exc:
                st.error(
                    "The backend could not process this prediction."
                )

                if exc.response is not None:
                    try:
                        st.code(
                            str(exc.response.json()),
                            language="json",
                        )
                    except ValueError:
                        st.write(exc.response.text)

            except requests.RequestException as exc:
                st.error(
                    "Could not reach the prediction API. "
                    "Check that FastAPI is running."
                )
                st.caption(str(exc))




# --------------------------------------------------
# Display prediction results
# --------------------------------------------------

if "interaction_result" in st.session_state:
    result = st.session_state["interaction_result"]

    drug_by_id = {
        str(drug["drug_id"]): drug
        for drug in drugs
    }

    result_drug_a = st.session_state.get("interaction_drug_a")
    result_drug_b = st.session_state.get("interaction_drug_b")

    st.divider()
    st.header("Prediction Results")

    # Selected drug pair
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Drug A**")
        st.write(display_drug_name(result_drug_a, drug_by_id))

    with col_b:
        st.markdown("**Drug B**")
        st.write(display_drug_name(result_drug_b, drug_by_id))

    # Model 1: Severity prediction
    st.subheader("Model 1 — Interaction Severity")

    severity = result.get("severity")

    if severity is not None:
        sev_lower = str(severity).strip().lower()
        if sev_lower == "major":
            st.error(f"🔴 Predicted Severity: **{severity}** — High Risk Interaction")
        elif sev_lower == "minor":
            st.success(f"🟢 Predicted Severity: **{severity}** — Low Risk Interaction")
        elif sev_lower == "moderate":
            st.warning(f"🟠 Predicted Severity: **{severity}** — Moderate Risk Interaction")
        else:
            st.info(f"Predicted Severity: **{severity}**")
    else:
        st.warning("Severity information was not returned by the API.")

    # Model 2: Top 10 predicted symptoms
    st.subheader("Model 2 — Top 10 Predicted Symptoms")

    ranked_effects = result.get("ranked_adverse_effects")

    if ranked_effects is None:
        ranked_effects = result.get("adverse_effect_scores")

    if ranked_effects is None:
        ranked_effects = result.get("probabilities")

    rows = []

    if isinstance(ranked_effects, dict):
        for symptom, score in ranked_effects.items():
            if isinstance(score, (int, float)):
                rows.append({
                    "Symptom": str(symptom),
                    "Confidence Level": float(score) * 100,
                })

    elif isinstance(ranked_effects, list):
        for item in ranked_effects:
            if not isinstance(item, dict):
                continue

            symptom = (
                item.get("effect")
                or item.get("name")
                or item.get("label")
                or item.get("symptom")
            )

            score = item.get("probability")
            if score is None:
                score = item.get("score")
            if score is None:
                score = item.get("confidence")

            if symptom is not None and isinstance(score, (int, float)):
                rows.append({
                    "Symptom": str(symptom),
                    "Confidence Level": float(score) * 100,
                })

    if rows:
        symptoms_df = pd.DataFrame(rows)

        symptoms_df = (
            symptoms_df
            .drop_duplicates(subset=["Symptom"])
            .sort_values("Confidence Level", ascending=False)
            .head(10)
            .reset_index(drop=True)
        )

        symptoms_df.index = symptoms_df.index + 1
        symptoms_df.index.name = "Rank"

        symptoms_df["Confidence Level"] = (
            symptoms_df["Confidence Level"].map(
                lambda value: f"{value:.2f}%"
            )
        )

        st.dataframe(
            symptoms_df,
            use_container_width=True,
        )
    else:
        st.info(
            "No ranked symptom scores were returned by the API."
        )

    disclaimer = result.get("disclaimer")

    if disclaimer:
        st.caption(str(disclaimer))

    st.caption(
        "Research prototype only. Severity and symptom scores are "
        "model predictions, not medical advice or verified clinical "
        "probabilities. Do not use them to make treatment decisions."
    )

# --------------------------------------------------
# Embedded Official n8n Chat Widget (@n8n/chat)
# --------------------------------------------------
import streamlit.components.v1 as components

N8N_CHAT_WEBHOOK_URL = "https://motamer540.app.n8n.cloud/webhook/d0a17e2f-9886-4ecf-ac56-b3c5f5010218/chat"

components.html(
    f"""
    <script>
    (function() {{
        try {{
            const targetDoc = (window.parent && window.parent.document) ? window.parent.document : document;
            const targetWin = (window.parent && window.parent.window) ? window.parent.window : window;

            // Remove any old native popovers if present
            const oldPopover = targetDoc.querySelector('div[data-testid="stPopover"]');
            if (oldPopover) oldPopover.style.display = 'none';

            // Check if already initialized to prevent duplicate widgets on rerun
            if (targetDoc.getElementById('n8n-chat-initialized')) {{
                return;
            }}

            // 1. Create target container in parent document body
            let chatContainer = targetDoc.getElementById('n8n-chat');
            if (!chatContainer) {{
                chatContainer = targetDoc.createElement('div');
                chatContainer.id = 'n8n-chat';
                targetDoc.body.appendChild(chatContainer);
            }}

            // 2. Add n8n style.css
            if (!targetDoc.getElementById('n8n-chat-style')) {{
                const link = targetDoc.createElement('link');
                link.id = 'n8n-chat-style';
                link.rel = 'stylesheet';
                link.href = 'https://cdn.jsdelivr.net/npm/@n8n/chat/dist/style.css';
                targetDoc.head.appendChild(link);
            }}

            // 3. Add Custom Design & Aesthetic Styles
            if (!targetDoc.getElementById('n8n-custom-theme')) {{
                const style = targetDoc.createElement('style');
                style.id = 'n8n-custom-theme';
                style.textContent = `
                    :root {{
                        --chat--color-primary: #2563eb;
                        --chat--color-primary-shade-50: #1d4ed8;
                        --chat--color-primary-shade-100: #1e40af;
                        --chat--header--background: #1e293b;
                        --chat--body--background: #0f172a;
                        --chat--footer--background: #1e293b;
                        --chat--message--bot--background: #1e293b;
                        --chat--message--bot--color: #f8fafc;
                        --chat--message--user--background: #2563eb;
                        --chat--message--user--color: #ffffff;
                        --chat--input--background: #0f172a;
                        --chat--input--text-color: #f8fafc;
                        --chat--window--border-radius: 20px;
                        --chat--window--width: 400px;
                        --chat--window--height: 540px;
                        --chat--window--z-index: 99999999;
                    }}

                    #n8n-chat {{
                        position: fixed !important;
                        bottom: 24px !important;
                        right: 24px !important;
                        z-index: 99999999 !important;
                    }}

                    /* Exact Old Floating Button Style */
                    #n8n-chat .chat-window-toggle {{
                        width: 62px !important;
                        height: 62px !important;
                        border-radius: 50% !important;
                        background-color: #2563eb !important;
                        background-image: 
                            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'%3E%3Cpath d='M12 2C6.477 2 2 6.477 2 12c0 1.821.487 3.53 1.338 5L2.1 21.2a1 1 0 0 0 1.2 1.2l4.2-1.238A9.957 9.957 0 0 0 12 22c5.523 0 10-4.477 10-10S17.523 2 12 2zm-3 8a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3zm6 0a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3zm-6.2 5.5a4.5 4.5 0 0 0 6.4 0 .75.75 0 1 1 1.06 1.06 6 6 0 0 1-8.52 0 .75.75 0 1 1 1.06-1.06z'/%3E%3C/svg%3E"),
                            linear-gradient(135deg, #3b82f6 0%, #1d4ed8 50%, #1e40af 100%) !important;
                        background-repeat: no-repeat, no-repeat !important;
                        background-position: center, center !important;
                        background-size: 32px 32px, 100% 100% !important;
                        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.5), 0 2px 6px rgba(0, 0, 0, 0.2) !important;
                        border: 2.5px solid rgba(255, 255, 255, 0.45) !important;
                        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
                        position: relative !important;
                        outline: none !important;
                        display: flex !important;
                        align-items: center !important;
                        justify-content: center !important;
                    }}

                    #n8n-chat .chat-window-toggle:hover {{
                        transform: scale(1.1) translateY(-4px) !important;
                        box-shadow: 0 14px 35px rgba(37, 99, 235, 0.65), 0 4px 12px rgba(0, 0, 0, 0.25) !important;
                        background-image: 
                            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'%3E%3Cpath d='M12 2C6.477 2 2 6.477 2 12c0 1.821.487 3.53 1.338 5L2.1 21.2a1 1 0 0 0 1.2 1.2l4.2-1.238A9.957 9.957 0 0 0 12 22c5.523 0 10-4.477 10-10S17.523 2 12 2zm-3 8a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3zm6 0a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3zm-6.2 5.5a4.5 4.5 0 0 0 6.4 0 .75.75 0 1 1 1.06 1.06 6 6 0 0 1-8.52 0 .75.75 0 1 1 1.06-1.06z'/%3E%3C/svg%3E"),
                            linear-gradient(135deg, #60a5fa 0%, #2563eb 50%, #1d4ed8 100%) !important;
                    }}

                    /* Hide n8n's default toggle SVG so our SVG robot face displays cleanly */
                    #n8n-chat .chat-window-toggle svg,
                    #n8n-chat .chat-window-toggle .chat-icon {{
                        opacity: 0 !important;
                        display: none !important;
                    }}

                    /* Green pulsing online badge on button */
                    #n8n-chat .chat-window-toggle::after {{
                        content: "" !important;
                        position: absolute !important;
                        top: 2px !important;
                        right: 2px !important;
                        width: 14px !important;
                        height: 14px !important;
                        background-color: #10b981 !important;
                        border: 2.5px solid #ffffff !important;
                        border-radius: 50% !important;
                        box-shadow: 0 0 8px rgba(16, 185, 129, 0.7) !important;
                        animation: afify-pulse 2s infinite !important;
                        z-index: 99 !important;
                        pointer-events: none !important;
                    }}

                    @keyframes afify-pulse {{
                        0% {{ box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }}
                        70% {{ box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }}
                        100% {{ box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }}
                    }}

                    /* Window border, shadow and header */
                    #n8n-chat .chat-window {{
                        border: 1px solid rgba(255, 255, 255, 0.15) !important;
                        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6) !important;
                    }}

                    #n8n-chat .chat-header h1::before {{
                        content: "🤖 ";
                        margin-right: 4px;
                    }}
                `;
                targetDoc.head.appendChild(style);
            }}

            // 4. Load UMD bundle and initialize createChat
            function initChat() {{
                if (targetWin.N8nChat && typeof targetWin.N8nChat.createChat === 'function') {{
                    const flag = targetDoc.createElement('div');
                    flag.id = 'n8n-chat-initialized';
                    targetDoc.body.appendChild(flag);

                    targetWin.N8nChat.createChat({{
                        webhookUrl: '{N8N_CHAT_WEBHOOK_URL}',
                        target: '#n8n-chat',
                        mode: 'window',
                        showWelcomeScreen: false,
                        defaultLanguage: 'en',
                        initialMessages: [
                            'Welcome to our Website👋\\n\\nMy name is HAKIM\\n\\nHow can I assist you today?'
                        ],
                        i18n: {{
                            en: {{
                                title: 'HAKIM',
                                subtitle: '',
                                footer: '',
                                getStarted: 'New Conversation',
                                inputPlaceholder: 'Type your question here...',
                            }}
                        }}
                    }});
                }}
            }}

            if (targetWin.N8nChat) {{
                initChat();
            }} else {{
                const script = targetDoc.createElement('script');
                script.id = 'n8n-chat-bundle';
                script.src = 'https://cdn.jsdelivr.net/npm/@n8n/chat/dist/chat.bundle.umd.js';
                script.onload = initChat;
                targetDoc.head.appendChild(script);
            }}
        }} catch (err) {{
            console.error('Error embedding n8n chat:', err);
        }}
    }})();
    </script>
    """,
    height=0,
    width=0,
)








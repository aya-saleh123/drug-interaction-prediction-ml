
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

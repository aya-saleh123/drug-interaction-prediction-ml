"""Inference pipeline for severity and 50-label adverse-effect models.

Feature construction mirrors the original notebook.
The PRR/regression model is intentionally excluded.
"""

from __future__ import annotations

import html
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

INLINE_TAG_PATTERN = re.compile(
    r"</?(?:sub|sup|i|b|em|strong|u)\s*/?>", re.IGNORECASE
)
ANY_TAG_PATTERN = re.compile(r"</?[A-Za-z][^<>]*>")
CITATION_TOKEN = r"(?:[A-Z]\d{1,8}|FDA [Ll]abel|Label|MSDS)"
CITATION_PATTERN = re.compile(
    r"\s*\[" + CITATION_TOKEN + r"(?:\s*,\s*" + CITATION_TOKEN + r")*\]"
)
WHITESPACE_PATTERN = re.compile(r"\s+")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required metadata file not found: {path}"
        )
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def clean_text_value(value: Any) -> Any:
    """Apply the configured text-cleaning steps."""
    if pd.isna(value):
        return np.nan

    text = INLINE_TAG_PATTERN.sub("", str(value))
    text = ANY_TAG_PATTERN.sub(" ", text)
    text = html.unescape(text)
    text = CITATION_PATTERN.sub("", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text if text else np.nan


def clean_category_value(value: Any) -> Any:
    """Normalize comma-separated categorical lists."""
    if pd.isna(value):
        return np.nan

    text = re.sub(r"\s*,\s*", ",", str(value).strip())
    return text if text else np.nan


class DrugInteractionPredictor:
    def __init__(
        self, artifacts_dir: Path = ARTIFACTS_DIR
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)

        self.api_schema = _read_json(
            self.artifacts_dir / "api_schema.json"
        )
        self.column_roles = _read_json(
            self.artifacts_dir / "column_roles.json"
        )
        self.preprocessing_config = _read_json(
            self.artifacts_dir / "preprocessing_config.json"
        )
        self.target_definitions = _read_json(
            self.artifacts_dir / "target_definitions.json"
        )
        self.severity_final = _read_json(
            self.artifacts_dir / "severity_final.json"
        )

        self.vectorizer = joblib.load(
            self.artifacts_dir / "tfidf_vectorizer.joblib"
        )
        self.preprocessor = joblib.load(
            self.artifacts_dir / "structured_preprocessor.joblib"
        )
        self.severity_model = joblib.load(
            self.artifacts_dir / "severity_model.joblib"
        )
        self.multilabel_model = joblib.load(
            self.artifacts_dir / "m3_logistic_regression.joblib"
        )

        # -----------------------------------------------------
        # Load the original catalog used by the ML pipeline.
        # Do not replace this catalog with the lookup CSV.
        # -----------------------------------------------------
        catalog_path = self.artifacts_dir / "drug_catalog.csv"

        if not catalog_path.exists():
            raise FileNotFoundError(
                f"Required drug catalog not found: {catalog_path}"
            )

        self.catalog = pd.read_csv(catalog_path)

        if "drug_id" not in self.catalog.columns:
            raise ValueError(
                "drug_catalog.csv must contain a 'drug_id' column."
            )

        self.catalog["drug_id"] = (
            self.catalog["drug_id"].astype(str).str.strip()
        )

        self.catalog_by_id = self.catalog.set_index(
            "drug_id", drop=False
        )

        # -----------------------------------------------------
        # Load the separate lookup used for display names.
        # This file does not supply features to the models.
        # -----------------------------------------------------
        lookup_path = self.artifacts_dir / "drug_lookup.csv"

        if not lookup_path.exists():
            raise FileNotFoundError(
                f"Required drug lookup not found: {lookup_path}"
            )

        self.drug_lookup = pd.read_csv(
            lookup_path,
            dtype={"drug_id": str},
        )

        required_columns = {"drug_id", "drug_name"}
        missing_columns = required_columns - set(
            self.drug_lookup.columns
        )

        if missing_columns:
            raise ValueError(
                "drug_lookup.csv is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        self.drug_lookup["drug_id"] = (
            self.drug_lookup["drug_id"].astype(str).str.strip()
        )
        self.drug_lookup["drug_name"] = (
            self.drug_lookup["drug_name"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        if self.drug_lookup["drug_id"].duplicated().any():
            raise ValueError(
                "drug_lookup.csv contains duplicate drug IDs."
            )

        if self.drug_lookup["drug_id"].eq("").any():
            raise ValueError(
                "drug_lookup.csv contains an empty drug ID."
            )

        # Ensure every lookup ID exists in the original catalog.
        unknown_ids = (
            set(self.drug_lookup["drug_id"])
            - set(self.catalog_by_id.index)
        )

        if unknown_ids:
            raise ValueError(
                "drug_lookup.csv contains IDs absent from "
                f"drug_catalog.csv: {sorted(unknown_ids)[:10]}"
            )

        # Use drug IDs as keys, never medicine names.
        self.lookup_by_id = self.drug_lookup.set_index(
            "drug_id"
        )

        self.pair_fields = self.api_schema["pair_input_fields"]
        self.text_columns = self.column_roles["text_columns"]
        self.smiles_columns = self.column_roles["smiles_columns"]
        self.category_columns = self.column_roles[
            "categorical_list_columns"
        ]
        self.numeric_columns = self.column_roles.get(
            "native_numeric_columns", []
        )
        self.missing_indicator_source_columns = (
            self.column_roles.get(
                "missing_indicator_source_columns", []
            )
        )

        fingerprint_config = self.preprocessing_config.get(
            "fingerprints", {}
        )
        self.fingerprints_created = bool(
            fingerprint_config.get("created", False)
        )
        self.fingerprint_bits = int(
            fingerprint_config.get("bits_per_drug", 2048)
        )

        if self.fingerprints_created:
            try:
                from rdkit import Chem, RDLogger
                from rdkit.Chem import rdFingerprintGenerator
            except ImportError as exc:
                raise RuntimeError(
                    "RDKit is required because the trained models use "
                    "molecular fingerprints. Install it with: "
                    "pip install rdkit"
                ) from exc

            RDLogger.DisableLog("rdApp.*")
            self.Chem = Chem
            self.fingerprint_generator = (
                rdFingerprintGenerator.GetMorganGenerator(
                    radius=2,
                    fpSize=self.fingerprint_bits,
                )
            )
        else:
            self.Chem = None
            self.fingerprint_generator = None

        self.severity_classes = (
            self.api_schema.get("outputs", {})
            .get("severity", {})
            .get("classes", ["Minor", "Moderate", "Major"])
        )

        self.adverse_event_definitions = (
            self.api_schema.get("outputs", {})
            .get("adverse_events", [])
        )

        if len(self.adverse_event_definitions) != 50:
            raise ValueError(
                "Expected exactly 50 adverse-event definitions in "
                "api_schema.json; found "
                f"{len(self.adverse_event_definitions)}."
            )

    def _get_catalog_drug(self, drug_id: str) -> pd.Series:
        drug_id = str(drug_id)

        if drug_id not in self.catalog_by_id.index:
            raise ValueError(f"Unknown drug_id: {drug_id}")

        return self.catalog_by_id.loc[drug_id]

    @staticmethod
    def _short_value(
        value: Any, limit: int = 90
    ) -> str:
        if pd.isna(value) or not str(value).strip():
            return ""

        value = re.sub(r"\s+", " ", str(value)).strip()

        if len(value) <= limit:
            return value

        return value[: limit - 1] + "…"

    def _drug_name(self, drug_id: str) -> str:
        """Return the display name associated with a catalog ID."""
        drug_id = str(drug_id)

        if drug_id not in self.lookup_by_id.index:
            return drug_id

        name = str(
            self.lookup_by_id.at[drug_id, "drug_name"]
        ).strip()

        return name or drug_id

    def _drug_label(self, row: pd.Series) -> str:
        """Return a readable label for prediction results."""
        drug_id = str(row["drug_id"])
        return self._drug_name(drug_id)

    def _drug_description(self, row: pd.Series) -> str:
        """Get a description from the original catalog."""
        for column in ("Indication", "Mechanism", "Toxicity"):
            if column in row.index:
                value = self._short_value(
                    row[column], limit=500
                )
                if value:
                    return value

        return "No description is available for this drug profile."

    def list_drugs(
        self,
        search: str | None = None,
        limit: int = 600,
    ) -> list[dict[str, str]]:
        """Return drug IDs, names, and descriptions for the UI."""
        frame = self.catalog

        if search and search.strip():
            needle = search.strip().casefold()

            searchable = [
                column
                for column in (
                    "drug_id",
                    "Indication",
                    "Mechanism",
                    "Toxicity",
                    "CYP450_Enzymes",
                )
                if column in frame.columns
            ]

            mask = pd.Series(False, index=frame.index)

            # Search the original catalog's text fields.
            for column in searchable:
                mask |= (
                    frame[column]
                    .fillna("")
                    .astype(str)
                    .str.casefold()
                    .str.contains(re.escape(needle), na=False)
                )

            # Also allow searching by medicine name.
            names = frame["drug_id"].map(
                self.lookup_by_id["drug_name"]
            ).fillna("")

            mask |= (
                names.astype(str)
                .str.casefold()
                .str.contains(re.escape(needle), na=False)
            )

            frame = frame[mask]

        frame = frame.head(max(1, min(int(limit), 600)))

        results: list[dict[str, str]] = []

        for _, row in frame.iterrows():
            drug_id = str(row["drug_id"])
            drug_name = self._drug_name(drug_id)

            results.append(
                {
                    "drug_id": drug_id,
                    "drug_name": drug_name,
                    "description": self._drug_description(row),
                    # Keep label for compatibility with the old frontend.
                    "label": drug_name,
                }
            )

        return results

    def _make_raw_pair(
        self,
        drug_a: pd.Series,
        drug_b: pd.Series,
    ) -> pd.DataFrame:
        record: dict[str, Any] = {}

        for field in self.pair_fields:
            name = field["name"]

            if name.endswith("_A"):
                catalog_column = name[:-2]
                record[name] = drug_a.get(
                    catalog_column, np.nan
                )
            elif name.endswith("_B"):
                catalog_column = name[:-2]
                record[name] = drug_b.get(
                    catalog_column, np.nan
                )
            else:
                raise ValueError(
                    f"Unexpected unpaired model input field: {name}"
                )

        return pd.DataFrame([record])

    def _clean_pair(
        self, raw: pd.DataFrame
    ) -> pd.DataFrame:
        cleaned = raw.copy()

        for column in self.text_columns:
            cleaned[column] = raw[column].map(clean_text_value)

        for column in self.category_columns:
            cleaned[column] = raw[column].map(
                clean_category_value
            )

        return cleaned

    def _combined_text(
        self, cleaned: pd.DataFrame
    ) -> pd.Series:
        pieces: dict[str, pd.Series] = {}

        for suffix in ("A", "B"):
            side_columns = [
                column
                for column in self.text_columns
                if column.endswith(f"_{suffix}")
            ]

            side_parts = []

            for column in side_columns:
                label = column[
                    : -(len(suffix) + 1)
                ].replace("_", " ").lower()

                # FIX: convert each individual value to text before
                # concatenation. Keep missing values excluded.
                text_values = cleaned[column].map(
                    lambda value: (
                        f"{label}: {value}"
                        if pd.notna(value)
                        else np.nan
                    )
                )

                side_parts.append(text_values)

            if side_parts:
                joined = pd.concat(side_parts, axis=1)
                pieces[suffix] = joined.apply(
                    lambda row: " ".join(
                        str(value)
                        for value in row.dropna()
                    ),
                    axis=1,
                )
            else:
                pieces[suffix] = pd.Series(
                    "", index=cleaned.index, dtype=str
                )

        return (
            "[DRUG A] "
            + pieces["A"].astype(str)
            + " [DRUG B] "
            + pieces["B"].astype(str)
        )

    def _structured_input(
        self,
        raw: pd.DataFrame,
        cleaned: pd.DataFrame,
    ) -> sparse.csr_matrix:
        engineered = pd.DataFrame(index=raw.index)

        for suffix in ("A", "B"):
            tag = suffix.lower()

            smiles_column = next(
                (
                    column
                    for column in self.smiles_columns
                    if column.endswith(f"_{suffix}")
                ),
                None,
            )

            if smiles_column:
                engineered[f"smiles_length_{tag}"] = (
                    raw[smiles_column]
                    .fillna("")
                    .astype(str)
                    .str.len()
                )

            side_text = [
                column
                for column in self.text_columns
                if column.endswith(f"_{suffix}")
            ]

            if side_text:
                engineered[f"text_chars_{tag}"] = (
                    cleaned[side_text]
                    .apply(
                        lambda col: col.fillna("").str.len()
                    )
                    .sum(axis=1)
                )

                engineered[f"text_fields_present_{tag}"] = (
                    cleaned[side_text].notna().sum(axis=1)
                )

        if len(self.smiles_columns) == 2:
            first, second = self.smiles_columns
            engineered["same_smiles"] = (
                raw[first] == raw[second]
            ).astype(int)

        for column in self.missing_indicator_source_columns:
            engineered[f"missing_{column}"] = (
                cleaned[column].isna().astype(int)
            )

        parts = [engineered]

        if self.numeric_columns:
            parts.append(raw[self.numeric_columns])

        categories = (
            cleaned[self.category_columns].fillna("")
            if self.category_columns
            else pd.DataFrame(index=raw.index)
        )
        parts.append(categories)

        model_input = pd.concat(parts, axis=1)

        expected = list(
            self.column_roles.get(
                "structured_feature_names", []
            )
        )

        if expected:
            required_inputs = (
                set(self.numeric_columns)
                | set(self.category_columns)
            )
            missing = [
                name
                for name in required_inputs
                if name not in model_input.columns
            ]

            if missing:
                raise ValueError(
                    "Structured inputs are missing expected columns: "
                    f"{missing}"
                )

        transformed = self.preprocessor.transform(model_input)

        return sparse.csr_matrix(
            transformed
        ).astype(np.float32)

    def _fingerprints(
        self, raw: pd.DataFrame
    ) -> sparse.csr_matrix:
        if not self.fingerprints_created:
            return sparse.csr_matrix(
                (len(raw), 0), dtype=np.float32
            )

        blocks = []

        for column in self.smiles_columns:
            values = raw[column].tolist()

            matrix = np.zeros(
                (len(values), self.fingerprint_bits),
                dtype=np.uint8,
            )

            for index, smiles in enumerate(values):
                if pd.isna(smiles) or not str(smiles).strip():
                    continue

                molecule = self.Chem.MolFromSmiles(
                    str(smiles)
                )

                if molecule is not None:
                    matrix[index] = (
                        self.fingerprint_generator
                        .GetFingerprintAsNumPy(molecule)
                    )

            blocks.append(
                sparse.csr_matrix(
                    matrix, dtype=np.float32
                )
            )

        return sparse.hstack(
            blocks,
            format="csr",
            dtype=np.float32,
        )

    def predict_pair(
        self,
        drug_a_id: str,
        drug_b_id: str,
    ) -> dict[str, Any]:
        # IMPORTANT: prediction still uses the original catalog.
        drug_a = self._get_catalog_drug(drug_a_id)
        drug_b = self._get_catalog_drug(drug_b_id)

        raw = self._make_raw_pair(drug_a, drug_b)
        cleaned = self._clean_pair(raw)

        # Feature 1: TF-IDF.
        combined_text = self._combined_text(cleaned)
        tfidf = self.vectorizer.transform(
            combined_text.fillna("").astype(str)
        ).astype(np.float32)

        # Feature 2: structured features.
        structured = self._structured_input(raw, cleaned)

        # Feature 3: molecular fingerprints.
        fingerprints = self._fingerprints(raw)

        # Severity model feature selection from severity_final.json.
        severity_feature_set = self.severity_final.get("features")

        if severity_feature_set == "fingerprints":
            severity_features = fingerprints

        elif severity_feature_set == "all_sources":
            # Notebook severity feature order:
            # TF-IDF, fingerprints, structured.
            severity_features = sparse.hstack(
                [tfidf, fingerprints, structured],
                format="csr",
                dtype=np.float32,
            )
        else:
            raise ValueError(
                "Unexpected 'features' value in severity_final.json: "
                f"{severity_feature_set!r}. Expected 'fingerprints' "
                "or 'all_sources'."
            )

        # Multilabel training feature order:
        # TF-IDF, structured, fingerprints.
        multilabel_features = sparse.hstack(
            [tfidf, structured, fingerprints],
            format="csr",
            dtype=np.float32,
        )

        # Run both classifiers.
        severity_index = int(
            self.severity_model.predict(
                severity_features
            )[0]
        )

        if not 0 <= severity_index < len(self.severity_classes):
            raise ValueError(
                "Severity model returned unexpected class index "
                f"{severity_index}."
            )

        multilabel_prediction = (
            self.multilabel_model.predict(
                multilabel_features
            )[0]
        )

        multilabel_probabilities = (
            self.multilabel_model.predict_proba(
                multilabel_features
            )[0]
        )

        print("Model type:", type(self.multilabel_model).__name__)
        print("Model classes:", self.multilabel_model.classes_)
        print("Prediction shape:", np.asarray(multilabel_prediction).shape)
        print(
            "Probability shape:",
            np.asarray(multilabel_probabilities).shape,
        )
        print(
            "Prediction dtype:",
            np.asarray(multilabel_prediction).dtype,
        )

        # =====================================================
        # TEMPORARY DIAGNOSTICS: inspect multilabel predictions.
        # =====================================================
        print("\n========== MULTILABEL DIAGNOSTICS ==========")
        print("Drug A ID:", drug_a_id)
        print("Drug B ID:", drug_b_id)

        print("TF-IDF shape:", tfidf.shape)
        print("TF-IDF nonzero values:", tfidf.nnz)

        print("Structured shape:", structured.shape)
        print("Structured nonzero values:", structured.nnz)

        print("Fingerprint shape:", fingerprints.shape)
        print("Fingerprint nonzero values:", fingerprints.nnz)

        print(
            "Combined multilabel shape:",
            multilabel_features.shape,
        )

        expected_features = getattr(
            self.multilabel_model,
            "n_features_in_",
            None,
        )
        print(
            "Model expected feature count:",
            expected_features,
        )

        print(
            "Number of positive predictions:",
            int(np.sum(multilabel_prediction)),
        )

        print(
            "Probability range:",
            float(np.min(multilabel_probabilities)),
            "to",
            float(np.max(multilabel_probabilities)),
        )

        print(
            "Probabilities >= 0.5:",
            int(np.sum(multilabel_probabilities >= 0.5)),
        )

        top_indices = np.argsort(
            multilabel_probabilities
        )[-10:][::-1]

        print("Top 10 adverse-effect scores:")

        for index in top_indices:
            definition = self.adverse_event_definitions[index]
            label = (
                definition.get("display_name")
                or definition["binary_column"]
            )

            print(
                f"  {label}: "
                f"{multilabel_probabilities[index]:.4f}"
            )

        print("============================================\n")
        # =====================================================

        events = []

        for index, definition in enumerate(
            self.adverse_event_definitions
        ):
            name = (
                definition.get("display_name")
                or definition["binary_column"]
            )

            events.append(
                {
                    "name": name,
                    "binary_column": definition["binary_column"],
                    "predicted": bool(
                        multilabel_prediction[index]
                    ),
                    "probability": float(
                        multilabel_probabilities[index]
                    ),
                }
            )

        ranked_events = sorted(
            events,
            key=lambda item: item["probability"],
            reverse=True,
        )

        return {
            "drug_a": {
                "drug_id": str(drug_a["drug_id"]),
                "label": self._drug_label(drug_a),
            },
            "drug_b": {
                "drug_id": str(drug_b["drug_id"]),
                "label": self._drug_label(drug_b),
            },
            "severity": self.severity_classes[severity_index],
            "predicted_adverse_effects": [
                event for event in events if event["predicted"]
            ],
            "ranked_adverse_effects": ranked_events,
            "disclaimer": (
                "Research/educational model output based on learned "
                "dataset associations. It does not establish that a "
                "drug pair causes an effect and is not medical advice."
            ),
        }


@lru_cache(maxsize=1)
def get_predictor() -> DrugInteractionPredictor:
    """Load models and artifacts once per Python process."""
    return DrugInteractionPredictor()
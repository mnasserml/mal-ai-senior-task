import os
import re
import logging
from typing import List

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from src.domain.ports import IPIIRedactor
from src.domain.models import PIIRedactionResult

logger = logging.getLogger(__name__)

class PresidioPIIAdapter(IPIIRedactor):
    def __init__(self, ner_model: str = "xx_ent_wiki_sm"):
        self.ner_model = ner_model
        self.analyzer = None
        self.anonymizer = None

        try:
            provider = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [
                    {"lang_code": "en", "model_name": "xx_ent_wiki_sm"},
                    {"lang_code": "ar", "model_name": "xx_ent_wiki_sm"}
                ]
            })
            nlp_engine = provider.create_engine()
            self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
            self.anonymizer = AnonymizerEngine()
            logger.info("Presidio loaded multilingual spaCy model: xx_ent_wiki_sm")
        except Exception as e:
            logger.warning("Falling back to default Presidio Analyzer Engine: %s", e)
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()

        self._add_custom_recognizers()

    def _add_custom_recognizers(self):
        # 1. Emirates ID Recognizer (784-YYYY-XXXXXXX-X)
        emirates_id_pattern = Pattern(
            name="emirates_id_pattern",
            regex=r'784-\d{4}-\d{7}-\d{1}',
            score=0.95
        )
        emirates_id_recognizer = PatternRecognizer(
            supported_entity="EMIRATES_ID",
            patterns=[emirates_id_pattern]
        )
        self.analyzer.registry.add_recognizer(emirates_id_recognizer)

        # 2. UAE IBAN Recognizer (AE + 21 digits)
        uae_iban_pattern = Pattern(
            name="uae_iban_pattern",
            regex=r'AE\d{21}',
            score=0.95
        )
        uae_iban_recognizer = PatternRecognizer(
            supported_entity="UAE_IBAN",
            patterns=[uae_iban_pattern]
        )
        self.analyzer.registry.add_recognizer(uae_iban_recognizer)

        # 3. Mal Account Number Recognizer (10 to 14 digits)
        account_num_pattern = Pattern(
            name="account_num_pattern",
            regex=r'\b\d{10,14}\b',
            score=0.85
        )
        account_num_recognizer = PatternRecognizer(
            supported_entity="ACCOUNT_NUMBER",
            patterns=[account_num_pattern]
        )
        self.analyzer.registry.add_recognizer(account_num_recognizer)

        # 4. UAE Phone Recognizer (+971 or 05x)
        uae_phone_pattern = Pattern(
            name="uae_phone_pattern",
            regex=r'(\+971|05)\d{8}',
            score=0.90
        )
        uae_phone_recognizer = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[uae_phone_pattern]
        )
        self.analyzer.registry.add_recognizer(uae_phone_recognizer)

        # 5. Arabic Person Name Pattern Recognizer (Conversational Intros & Titles)
        arabic_name_patterns = [
            Pattern(
                name="arabic_greeting_name",
                regex=r'(?:معك|معكم|أنا|اسمى|اسمي|أخوكم|أختكم|العميل)\s+([\u0621-\u064A]+(?:\s+[\u0621-\u064A]+){0,3})',
                score=0.90
            ),
            Pattern(
                name="arabic_title_name",
                regex=r'(?:السيد|السيدة|الدكتور|الدكتورة|المهندس|المهندسة|أ\.|د\.|م\.)\s+([\u0621-\u064A]+(?:\s+[\u0621-\u064A]+){0,3})',
                score=0.90
            )
        ]
        self.analyzer.registry.add_recognizer(
            PatternRecognizer(supported_entity="PERSON", patterns=arabic_name_patterns, supported_language="ar")
        )
        self.analyzer.registry.add_recognizer(
            PatternRecognizer(supported_entity="PERSON", patterns=arabic_name_patterns, supported_language="en")
        )

    def redact(self, text: str, language: str = "en") -> PIIRedactionResult:
        # 1. Analyze text with Presidio using multilingual spaCy model
        results_en = self.analyzer.analyze(
            text=text,
            entities=[
                "PERSON", "PER", "PHONE_NUMBER", "EMAIL_ADDRESS",
                "EMIRATES_ID", "UAE_IBAN", "ACCOUNT_NUMBER"
            ],
            language="en"
        )
        results_ar = self.analyzer.analyze(
            text=text,
            entities=[
                "PERSON", "PER", "PHONE_NUMBER", "EMAIL_ADDRESS",
                "EMIRATES_ID", "UAE_IBAN", "ACCOUNT_NUMBER"
            ],
            language="ar"
        )

        all_results = results_en + results_ar
        spans_to_replace = []

        for res in all_results:
            entity_type = "PERSON" if res.entity_type in ["PERSON", "PER"] else res.entity_type
            spans_to_replace.append({
                "start": res.start,
                "end": res.end,
                "replacement": f"<{entity_type}>",
                "entity_type": entity_type,
                "score": res.score
            })

        # 2. Arabic Name Conversational Pattern Matching
        arabic_intro_regex = r'(اسمي|أنا|معك|معكم|العميل)\s+([\u0621-\u064A]+(?:\s+(?:بن|آل|عبد\s+[\u0621-\u064A]+|[\u0621-\u064A]+)){1,3})'
        for match in re.finditer(arabic_intro_regex, text):
            name_start = match.start(2)
            name_end = match.end(2)
            spans_to_replace.append({
                "start": name_start,
                "end": name_end,
                "replacement": "<PERSON>",
                "entity_type": "PERSON",
                "score": 0.90
            })

        # 3. Filter overlapping spans and sort
        spans_to_replace.sort(key=lambda s: s['start'])
        non_overlapping_spans = []
        last_end = -1
        for span in spans_to_replace:
            if span['start'] >= last_end:
                non_overlapping_spans.append(span)
                last_end = span['end']

        # 4. Apply character replacements on original text from right to left
        final_text = text
        detected_entities = []
        for span in sorted(non_overlapping_spans, key=lambda s: s['start'], reverse=True):
            s_start, s_end = span['start'], span['end']
            final_text = final_text[:s_start] + span['replacement'] + final_text[s_end:]
            detected_entities.append({
                "entity_type": span['entity_type'],
                "start": s_start,
                "end": s_end,
                "score": span['score']
            })

        return PIIRedactionResult(
            original_text=text,
            redacted_text=final_text,
            detected_entities=detected_entities,
            redacted_count=len(non_overlapping_spans)
        )

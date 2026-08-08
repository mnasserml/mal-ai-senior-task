import re
import logging
from typing import List

from apii import default_pipeline
from src.domain.ports import IPIIRedactor
from src.domain.models import PIIRedactionResult

logger = logging.getLogger(__name__)

class AajilPIIAdapter(IPIIRedactor):
    """
    Lightweight, production-grade Arabic & GCC PII Redactor adapter powered by
    Aajil-Labs int8 ONNX engine (apii). Zero PyTorch overhead (~470MB RAM, <10ms inference).
    """

    def __init__(self):
        try:
            self.pipeline = default_pipeline()
            logger.info("Loaded Aajil-Labs ONNX PII pipeline (apii)")
        except Exception as e:
            logger.error("Failed to initialize Aajil PII pipeline: %s", e)
            self.pipeline = None

        # Custom Account Number Recognizer (10 to 14 digits)
        self.account_num_regex = re.compile(r'\b\d{10,14}\b')

    def redact(self, text: str, language: str = "en") -> PIIRedactionResult:
        if not self.pipeline:
            return PIIRedactionResult(
                original_text=text,
                redacted_text=text,
                detected_entities=[],
                redacted_count=0
            )

        detections = self.pipeline.detect(text)
        spans_to_replace = []

        # 1. Process ONNX model detections
        for d in detections:
            kind_str = d.kind.name.upper()
            if kind_str in ['PERSON', 'NAME']:
                entity_type = 'PERSON'
            elif kind_str in ['NATIONAL_ID', 'NAT_ID']:
                entity_type = 'EMIRATES_ID'
            elif kind_str == 'IBAN':
                entity_type = 'UAE_IBAN'
            elif kind_str == 'PHONE':
                entity_type = 'PHONE_NUMBER'
            elif kind_str == 'EMAIL':
                entity_type = 'EMAIL_ADDRESS'
            elif kind_str in ['ACCOUNT', 'ACCOUNT_NUMBER']:
                entity_type = 'ACCOUNT_NUMBER'
            else:
                entity_type = kind_str

            spans_to_replace.append({
                'start': d.start,
                'end': d.end,
                'replacement': f'<{entity_type}>',
                'entity_type': entity_type,
                'score': d.confidence
            })

        # 2. UAE IBAN pattern fallback (AE + 21 digits)
        iban_regex = re.compile(r'AE\d{21}')
        for match in iban_regex.finditer(text):
            m_start, m_end = match.start(), match.end()
            if not any(s['start'] <= m_start and m_end <= s['end'] for s in spans_to_replace):
                spans_to_replace.append({
                    'start': m_start,
                    'end': m_end,
                    'replacement': '<UAE_IBAN>',
                    'entity_type': 'UAE_IBAN',
                    'score': 0.95
                })

        # 3. Account Number pattern fallback (10 to 14 digits)
        for match in self.account_num_regex.finditer(text):
            m_start, m_end = match.start(), match.end()
            # Avoid replacing if already captured by IBAN or Emirates ID
            if not any(s['start'] <= m_start and m_end <= s['end'] for s in spans_to_replace):
                spans_to_replace.append({
                    'start': m_start,
                    'end': m_end,
                    'replacement': '<ACCOUNT_NUMBER>',
                    'entity_type': 'ACCOUNT_NUMBER',
                    'score': 0.85
                })

        # 3. Filter overlapping spans & sort
        spans_to_replace.sort(key=lambda s: s['start'])
        non_overlapping_spans = []
        last_end = -1
        for span in spans_to_replace:
            if span['start'] >= last_end:
                non_overlapping_spans.append(span)
                last_end = span['end']

        # 4. Perform character replacements right to left
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

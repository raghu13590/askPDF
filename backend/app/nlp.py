import spacy

# Load at module import (cached in container)
_nlp = spacy.load("en_core_web_sm")

def split_into_sentences(text: str):
    doc = _nlp(text)
    sentences = []
    for i, sent in enumerate(doc.sents):
        s = sent.text.strip()
        if s:
            sentences.append({"id": i, "text": s})
    return sentences

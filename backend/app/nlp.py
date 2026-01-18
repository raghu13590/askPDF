
import spacy

_nlp = spacy.load("en_core_web_sm")

def split_into_sentences(text: str):
    """
    Splits the input text into sentences, preserving paragraph and header boundaries.
    
    The function first splits the text by double newlines to respect hard boundaries (such as headers and paragraphs),
    then uses spaCy to further split each chunk into sentences.
    
    Args:
        text (str): The input text to split.
    
    Returns:
        list[dict]: A list of dictionaries, each containing an 'id' and 'text' for each sentence.
    """
    chunks = text.split("\n\n")
    sentences = []
    global_id = 0
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        doc = _nlp(chunk)
        for sent in doc.sents:
            s = sent.text.strip()
            if s:
                sentences.append({"id": global_id, "text": s})
                global_id += 1
    return sentences

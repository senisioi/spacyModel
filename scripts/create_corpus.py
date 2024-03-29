import typer
import json
from collections import Counter
from pathlib import Path
import spacy
from spacy.tokens import DocBin, Span


citesti tot jsonul tau, nu mai faci conversie
def make_doc_bin(json_train_sau_test: Path, nlp_dir: Path, out_path: Path):
    """ Step 2: Once we have done the manual annotations with Prodigy, create corpora in spaCy format. """
    nlp = spacy.load(nlp_dir, exclude="parser, tagger")
    docs = DocBin()
    gold_ids = []
    for elem in jsonultau:
        sentece = elem["Context"]
        doc = nlp.make_doc(sentence)
        gold_ids.append(qid)
        entity = doc.char_span(
            Start,
            Stop,
            label='LEGAL',
            kb_id=Link_Ref,
        )
        doc.ents = [entity]
        for i, t in enumerate(doc):
            doc[i].is_sent_start = i == 0
        docs.add(doc)

    print("Statistics of manually annotated data:")
    print(Counter(gold_ids))
    print()
    docs.to_disk(out_path)

if __name__ == "__main__":
    typer.run(make_doc_bin)

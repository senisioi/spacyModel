import json
import os
from collections import Counter
from pathlib import Path

import spacy
from spacy.tokens import DocBin, Span


# Function to align character spans with tokens
def align_span_to_tokens(doc, start_char, end_char, label):
    start_token, end_token = None, None
    for token in doc:
        if start_char >= token.idx and (start_token is None or start_char < token.idx + len(token)):
            start_token = token.i
        if end_char <= token.idx + len(token.text) and (end_token is None or end_char > token.idx):
            end_token = token.i + 1
    if start_token is not None and end_token is not None and start_token < end_token:
        return Span(doc, start_token, end_token, label=label)
    else:
        return None

# citesti tot jsonul tau, nu mai faci conversie
def make_doc_bin(json_train_sau_test: Path, nlp_dir: Path, out_path: Path):
    # nlp = spacy.load(nlp_dir, exclude="parser, tagger")
    nlp = spacy.load(nlp_dir, exclude="parser, tagger")
    docs = DocBin()
    gold_ids = []
    failed_entities = 0

    # Load JSON data
    with open(json_train_sau_test, 'r', encoding='utf-8') as file:
        json_data = json.load(file)


    # Creare lista colectat entitati
    ent = []
    for elem in json_data:
        # sentence = elem["Context"]
        sentence =elem["Context"]
        # print(sentence)
        doc = nlp.make_doc(sentence)
        # print(doc)
        # print()
        start = elem["Start"]
        if start <0 :
            continue
        stop = elem["Stop"]
        link_ref = elem["Link_Ref"]
        gold_ids.append(link_ref)
        # gold_ids.append(link_ref)
        # entity = doc.char_span(
        #     start,
        #     stop,
        #     label = 'LEGAL',
        #     kb_id = link_ref,
        #     alignment_mode='contract'
        # )

        entity =align_span_to_tokens(doc, start, stop, "LEGAL")
        if entity is not None:
            doc.ents = [entity]
        else:
            failed_entities +=1
            print(f"Context: {sentence}")
            print(f"Entity from JSON file {elem['Mention']}")
            print(f"Failed to create entity from {start} to {stop} in context: {sentence[start:stop]}")
            print(f"Failed to create entity {link_ref}")
            print()
            continue
        # doc.ents = ents

        #
        for i, token in enumerate(doc):
            doc[i].is_sent_start = i == 0
        docs.add(doc)
        # print(doc)
        # print()

    print("Statistics of manually annotated data:")
    print(Counter(gold_ids))
    print(failed_entities)
    docs.to_disk(out_path)


if __name__ == "__main__":
    print(os.getcwd())
    make_doc_bin(
         json_train_sau_test=Path("all_datas_training_v2.json"),
         nlp_dir=Path("../temp/my_nlp"),
         out_path=Path("../corpus/dev_train.spacy")
     )
    make_doc_bin(
        json_train_sau_test=Path("all_datas_test_v2.json"),
        nlp_dir=Path("../temp/my_nlp"),
        out_path=Path("../corpus/dev_test.spacy")
    )
    # typer.run(make_doc_bin)

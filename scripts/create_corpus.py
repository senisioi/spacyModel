import json
import os
from collections import Counter
from pathlib import Path

import spacy
from spacy.tokens import DocBin, Span


def make_doc_bin(json_train_sau_test: Path, nlp_dir: Path, out_path: Path):
    nlp = spacy.load(nlp_dir, exclude="parser, tagger")
    docs = DocBin()
    gold_ids = []
    failed_entities = 0

    # Load JSON data
    with open(json_train_sau_test, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    for elem in json_data:
        sentence =elem["Context"]
        doc = nlp.make_doc(sentence)
        start = elem["Start"]
        if start <0 :
            continue
        stop = elem["Stop"]
        link_ref = elem["Link_Ref"]
        gold_ids.append(link_ref)
        entity = doc.char_span(
            start,
            stop,
            label = 'LEGAL',
            alignment_mode='expand',
            kb_id = link_ref
        )

        # entity =align_span_to_tokens(doc, start, stop, kb_id=link_ref, label="LEGAL")
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

        #
        for i, token in enumerate(doc):
            doc[i].is_sent_start = i == 0
        docs.add(doc)
        # print(doc)
        # print()

    print("Statistics of manually annotated data:")
    print(Counter(gold_ids))
    print(failed_entities)
    print()
    # print("Statistics of manually annotated data:", len(gold_ids))
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

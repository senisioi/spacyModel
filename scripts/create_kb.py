import typer
import csv
import os
from pathlib import Path

import spacy
from spacy.kb import InMemoryLookupKB


def main(entities_loc: Path, vectors_model: str, kb_loc: Path, nlp_dir: Path):
    """ Step 1: create the Knowledge Base in spaCy and write it to file """

    # First: create a simpel model from a model with an NER component
    # To ensure we get the correct entities for this demo, add a simple entity_ruler as well.
    nlp = spacy.load(vectors_model, exclude="parser, tagger, lemmatizer")
    ruler = nlp.add_pipe("entity_ruler", after="ner")
    patterns = [{"label": "PERSON", "pattern": [{"LOWER": "emerson"}]}]
    ruler.add_patterns(patterns)
    nlp.add_pipe("sentencizer", first=True)

    name_dict, desc_dict = _load_entities(entities_loc)

    kb = InMemoryLookupKB(vocab=nlp.vocab, entity_vector_length=280)

    # TODO:
    iei toate titlurile unice
    adaugi Lin_Ref coresp fiecarui titlu

    #kb.add_entity(entity=link_ref, entity_vector=desc_enc, freq=342)
    for qid, titlu in name_dict.items():
        desc_doc = nlp(titlu)
        desc_enc = desc_doc.vector
        # Set arbitrary value for frequency
        kb.add_entity(entity=qid, entity_vector=desc_enc, freq=342)

    iei toate mentiunile unice si pentru fiecare mentiune
    faci o lista cu toate Link_Ref (qid)
    mentiune
    lista_de_link_ref_pt_mentiune
    ctr = Counter(lista_de_link_ref_pt_mentiune)
    frecv = np.array(list(ctr.values()))
    proba = frecv/frecv.sum()
    kb.add_alias(alias=mentiune, entities=lista_de_link_ref_pt_mentiune, probabilities= proba)
    for qid, name in name_dict.items():
        # set 100% prior probability P(entity|alias) for each unique name
        kb.add_alias(alias=mentiune, entities=[qid], probabilities=[1])


    print(f"Entities in the KB: {kb.get_entity_strings()}")
    print(f"Aliases in the KB: {kb.get_alias_strings()}")
    print()
    kb.to_disk(kb_loc)
    if not os.path.exists(nlp_dir):
        os.mkdir(nlp_dir)
    nlp.to_disk(nlp_dir)


def _load_entities(entities_loc: Path):
    """ Helper function to read in the pre-defined entities we want to disambiguate to. """
    names = dict()
    descriptions = dict()
    with entities_loc.open("r", encoding="utf8") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        for row in csvreader:
            qid = row[0]
            name = row[1]
            desc = row[2]
            names[qid] = name
            descriptions[qid] = desc
    return names, descriptions


if __name__ == "__main__":
    print(os.getcwd())
    main(entities_loc=Path("../assets/entities.csv"), vectors_model="ro_legal_fl", kb_loc="/temp/my_kb",
         nlp_dir="/temp/nlp_dir")
   # typer.run(main)

import csv
import json
import os
from collections import OrderedDict, Counter
from pathlib import Path

import numpy as np
import spacy
from spacy.kb import InMemoryLookupKB


def buildEntities(cale_fisier1 = "../scripts/all_datas_test_v2.json", cale_fisier2 = "../scripts/all_datas_training_v2.json", cale_fisier_iesire="../assets/entities.csv"):
    data = []

    for cale_fisier in [cale_fisier1, cale_fisier2]:
        with open(cale_fisier, 'r', encoding='utf-8') as fisier:
            data.extend(json.load(fisier))
    linkuri_unice = OrderedDict()

    for element in data:
        link_ref = element.get('Link_Ref', '')
        # Verificăm dacă linkul a fost deja adăugat pentru a asigura unicitatea
        if link_ref not in linkuri_unice:
            linkuri_unice[link_ref] = f'"{link_ref}","{element["Titlu_Ref_Clean"]}","{element["Titlu_link_Ref"]}"\n'

    with open(cale_fisier_iesire, 'w', encoding='utf-8') as fisier_iesire:
        for linie in linkuri_unice.values():
            fisier_iesire.write(linie)


def buildMentionsAmbiguity():
    data = []
    for cale_fisier in ['all_datas_training_v2.json', 'all_datas_test_v2.json']:
        with open(cale_fisier, 'r', encoding='utf-8') as fisier:
            data.extend(json.load(fisier))

    mentions_dict = {}
    for item in data:
        mention = item['Mention']
        link_ref = item['Link_Ref']
        mentions_dict.setdefault(mention, set()).add(link_ref)

    #  Convert Sets to Lists
    for mention in mentions_dict:
        mentions_dict[mention] = list(mentions_dict[mention])

    # print(mentions_dict)
    return mentions_dict

    # Write to a JSON File
    # with open('mentions_to_links.json', 'w', encoding='utf-8') as outfile:
    #     json.dump(mentions_dict, outfile, ensure_ascii=False, indent=4)


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
    # iei toate titlurile unice
    # adaugi Lin_Ref coresp fiecarui titlu
    # DONE

    # kb.add_entity(entity=link_ref, entity_vector=desc_enc, freq=342)
    for link, titlu in name_dict.items():
        desc_doc = nlp(titlu)
        desc_enc = desc_doc.vector
        # Set arbitrary value for frequency
        kb.add_entity(entity=link, entity_vector=desc_enc, freq=280)

    # iei toate mentiunile unice si pentru fiecare mentiune
    # faci o lista cu toate Link_Ref (qid)
    # mentiune
    # lista_de_link_ref_pt_mentiune
    mention_dict = buildMentionsAmbiguity()
    print(mention_dict)
    for mention, links_list in mention_dict.items():
        ctr = Counter(links_list)
        frecv = np.array(list(ctr.values()))
        probability = frecv / frecv.sum()
        kb.add_alias(alias=mention, entities=links_list, probabilities=probability)
    for qid, name in name_dict.items():
        # set 100% prior probability P(entity|alias) for each unique name
        kb.add_alias(alias=mention, entities=[qid], probabilities=[1])

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
    buildEntities( "all_datas_test_v2.json", "all_datas_training_v2.json", "../assets/entities.csv")
    main(entities_loc=Path("../assets/entities.csv"), vectors_model="ro_legal_fl", kb_loc="/temp/my_kb",
         nlp_dir="/temp/my_nlp")
# typer.run(main)

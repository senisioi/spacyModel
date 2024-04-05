import spacy

nlp = spacy.blank("ro")

spacy_file = '../corpus/dev_test.spacy'

doc_bin = spacy.tokens.DocBin().from_disk(spacy_file)
docs = list(doc_bin.get_docs(nlp.vocab))


for doc in docs:
    print("Text:", doc.text)  # documentului
    for ent in doc.ents:
        print(" Entitate:", ent.text, ent.label_, ent.kb_id_)  # textul, eticheta și ID-ul KB
        print()
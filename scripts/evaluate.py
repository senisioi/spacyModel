import typer
from pathlib import Path
import os

import spacy
from spacy.tokens import DocBin
from spacy.training import Example

# we need to import this to parse the custom reader from the config
from custom_functions import create_docbin_reader


def main(nlp_dir: Path, dev_set: Path):
    """ Step 4: Evaluate the new Entity Linking component by applying it to unseen text. """
    nlp = spacy.util.load_model_from_path(nlp_dir)
    examples = []

    correct_predictions = 0
    total_predictions = 0
    no_entity = 0
    with open(dev_set, "rb") as f:
        doc_bin = DocBin().from_disk(dev_set)
        docs = doc_bin.get_docs(nlp.vocab)
        for doc in docs:
            examples.append(Example(nlp(doc.text), doc))

    print("\nRESULTS ON THE DEV SET:")
    for example in examples:
        total_predictions += 1
        if example.predicted.ents:
            print(f"Gold annotation: {example.reference.ents[0].kb_id_}")
            print(f"Mention: {example.reference.ents[0]}")
            print(f"Predicted annotation: {example.predicted.ents[0].kb_id_}")
            if example.reference.ents[0].kb_id_ == example.predicted.ents[0].kb_id_:
                correct_predictions += 1
        else:
            # print(f"Example {example}")
            print("Gold Id" + str(example.reference.ents[0].kb_id_))
            print(f"Mention: {example.reference.ents[0]}")
            print("Predictie " + str(example.predicted.ents))
            no_entity +=1
        print()

    print()
    print("RUNNING THE PIPELINE ON UNSEEN TEXT:")
    accuracy = correct_predictions / (total_predictions)
    print("Correct predinctions  " + str(correct_predictions))
    print("Total predinctions  " + str(total_predictions))
    print("No entity  " + str(no_entity))
    print("Accuracy: " + str(accuracy))
    # text = "Tennis champion Emerson was expected to win Wimbledon."
    # doc = nlp(text)
    # print(text)
    # for ent in doc.ents:
    #     print(ent.text, ent.label_, ent.kb_id_)



if __name__ == "__main__":
    typer.run(main)

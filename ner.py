import os
from main_utils import import_attribution_doc
import en_core_web_lg
import string

nlp = en_core_web_lg.load()


def get_ne_info(corpus_directory):
    """
    Adds two columns to the existing DataFrame with
    (1) IOB tags that indicate whether the word is the B(eginning), I(nside) or O(utside) of Named Entity (NE).
    (2) information about the type of NE (e.g., person, organisation, date, etc)
    # TODO: Should I remove the irrelevant types such as dates, and only keep person, organisation, etc.
    """
    for file in os.listdir(corpus_directory):
        list_of_words = []
        df = import_attribution_doc(corpus_directory + file)
        for index, entry in df.iterrows():
            list_of_words.append(entry["word"])

        paragraph = "".join(
            [" " + i if not i.startswith("'") and i not in string.punctuation else i for i in list_of_words]).strip()
        doc = nlp(paragraph)

        iob_tags = []
        ent_types = []
        for word in doc:
            iob_tags.append(word.ent_iob_)
            ent_types.append(word.ent_type_)  # TODO: should I replace the empty values with O or another value

        df["iob"] = iob_tags
        df["ent"] = ent_types

        # TODO: Where to write the DataFrame to?
        return df


def run():
    corpus_directory = "../data/parc30-conll/train-conll-foreval/"
    get_ne_info(corpus_directory)


if __name__ == "__main__":
    run()

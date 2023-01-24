import pickle

from pycocotools.coco import COCO
from torchtext.vocab import GloVe

WORD_EMBEDDING_DIM = 300
ANNOTATION_FILE = "./data/coco/annotations/instances_train2017.json"

def main():
    """
    Our category mappings:
    {
       "stoi": {
            ...,
            "baseball": glove.stoi["baseball"],
            "bat": glove.stoi["bat"],
            ...
        }
    }
    Original author's category mappings:
    {
        "stoi": {
            ...,
            "baseball bat": [glove.stoi["baseball"],
                             glove.stoi["bat"]],
            ...   
        }
    }
    """
    glove = GloVe(name="6B", dim=WORD_EMBEDDING_DIM)
    coco = COCO(ANNOTATION_FILE)
    
    # stoi: Coco catgeory to GloVe index.
    # itos: GloVe index to Coco category.
    cat_mappings = {"stoi": {}, "itos": {}}
    for cat in coco.cats.values():
        for token in cat["name"].split(" "):
            cat_mappings["stoi"][token] = glove.stoi[token]
            cat_mappings["itos"][glove.stoi[token]] = token
            
    # Save the category mappings into a pickle file.
    with open("./data/coco/cat_mappings.pkl", "wb") as f:    
        pickle.dump(cat_mappings, f)
    

if __name__ == "__main__":
    main()
    
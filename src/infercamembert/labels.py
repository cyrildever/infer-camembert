class Labels:
    ALL_INDICES: list
    ALL_KEYS: list

    def __init__(self, dictionary: dict) -> None:
        self.dictionary = dictionary
        Labels.ALL_KEYS = dictionary.keys()
        Labels.ALL_INDICES = list(range(len(Labels.ALL_KEYS)))
    
    def get_label_from_dictionary(self, id):
        return self.dictionary[id]

    def get_id2key(self):
        return {k: l for k, l in enumerate(Labels.ALL_KEYS)}

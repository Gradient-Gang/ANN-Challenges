def getRaise(dictionary: dict, key: str, dictionaryName: str = "dictionary"):
    if key in dictionary:
        return dictionary[key]
    else:
        raise KeyError(f"Key '{key}' not found in {dictionaryName}.")

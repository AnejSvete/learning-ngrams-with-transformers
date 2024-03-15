import random
from typing import List, Tuple

from art_ngrams.lm_generation import ngram


def get_data(
    pLM: ngram.NgramLM, N_train: int, N_val: int, N_test: int, n: int, seed: int
) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:

    while True:

        possible_strings = list(set(pLM.sample(100000, to_string=True)))
        M = len(possible_strings)
        random.shuffle(possible_strings)

        train_strings = set(possible_strings[: M // 3])
        val_strings = set(possible_strings[M // 3 : 2 * M // 3])
        test_strings = set(possible_strings[2 * M // 3 :])

        D_train = pLM.sample(100000, to_string=True)
        D_val = pLM.sample(100000, to_string=True)
        D_test = pLM.sample(100000, to_string=True)
        print(
            f"Sampled {len(D_train)} train, {len(D_val)} validation, "
            f"and {len(D_test)} test strings."
        )
        D_train = [x for x in D_train if x not in val_strings]
        D_train = [x for x in D_train if x not in test_strings]
        D_val = [x for x in D_val if x not in train_strings]
        D_val = [x for x in D_val if x not in test_strings]
        D_test = [x for x in D_test if x not in train_strings]
        D_test = [x for x in D_test if x not in val_strings]

        train_set = set(D_train)
        val_set = set(D_val)
        test_set = set(D_test)

        D_train = [x for x in D_train if x not in val_set]
        D_train = [x for x in D_train if x not in test_set][:N_train]
        D_val = [x for x in D_val if x not in train_set]
        D_val = [x for x in D_val if x not in test_set][:N_val]
        D_test = [x for x in D_test if x not in train_set]
        D_test = [x for x in D_test if x not in val_set][:N_test]

        if len(D_train) == N_train and len(D_val) == N_val and len(D_test) == N_test:
            break
        else:
            print(
                f"Retrying. Got {len(D_train)} train, {len(D_val)} validation, "
                f"and {len(D_test)} test strings."
            )

    print(
        f"Retained {len(D_train)} train, {len(D_val)} validation, "
        f"and {len(D_test)} test strings."
    )

    return D_train, D_val, D_test

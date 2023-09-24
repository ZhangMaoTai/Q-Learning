MAX_WORD_LEN = 32
MAX_TIME_STEP = 12
OUTPUT_DIM = 26
EMBEDDING_DIM = 10

STATE_MAPPING_INT_TO_STR = {
    **{0: "_", 1: "PAD"},
    **{i - 96 + 1: chr(i) for i in range(97, 123)}
}

STATE_MAPPING_STR_TO_INT = {
    v: k
    for k, v in STATE_MAPPING_INT_TO_STR.items()
}

ACTION_MAPPING_INT_TO_STR = {
    i - 96 - 1: chr(i) for i in range(97, 123)
}

ACTION_MAPPING_STR_TO_INT = {
    v: k
    for k, v in ACTION_MAPPING_INT_TO_STR.items()
}


MAX_TRY = 6

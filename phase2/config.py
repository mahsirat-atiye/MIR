class Config:
    NUM_OF_CLASSES = 4

    BODY = 'body'
    TITLE = 'title'
    CATEGORY = 'category'

    DATA_DIR = "data/"
    TRAIN_DATA = DATA_DIR + "train.json"
    VALIDATION_DATA = DATA_DIR + "validation.json"

    TOTAL_NUM_OF_TRAIN_DOCS = 24000
    TOTAL_NUM_OF_VALIDATION_DOCS = 3000

    TRAIN_TIME_SAVING = int(TOTAL_NUM_OF_TRAIN_DOCS/10)
    VALIDATION_TIME_SAVING = int(TOTAL_NUM_OF_VALIDATION_DOCS/10)






from enum import Enum

class QuestionType(str, Enum):
    YES_NO = "Yes/No"
    MULTIPLE_CHOICE = "Multiple choice"

class VectorStoreType(str, Enum):
    MONGO = "mongo"
    ELASTIC = "elastic"
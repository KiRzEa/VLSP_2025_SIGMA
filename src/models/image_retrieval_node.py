import torch
from pydantic import BaseModel
from typing import List, Dict, Any

class ImageEntry(BaseModel):
    filename: str
    metadata: Dict[str, Any]
    description: List[Dict[str, Any]]
    feature: torch.Tensor

    class Config:
        arbitrary_types_allowed = True  # to allow torch.Tensor

    def __str__(self):
        description_summary = (
            f"{len(self.description)} descriptions"
            if self.description else "no descriptions"
        )

        first_desc_keys = (
            list(self.description[0].keys())
            if self.description else []
        )

        return (
            f"ImageEntry(filename={self.filename!r}, "
            f"metadata_keys={list(self.metadata.keys())}, "
            f"{description_summary}, "
            f"first_description_keys={first_desc_keys})"
        )

    def __repr__(self):
        return self.__str__()
   
class SearchResultEntry(ImageEntry):
    score: float

    def __str__(self):
        base = super().__str__()  # use ImageEntry’s __str__
        return f"SearchResultEntry({base}, score={self.score:.4f})"

    def __repr__(self):
        return self.__str__()
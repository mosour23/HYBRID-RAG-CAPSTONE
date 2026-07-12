from abc import ABC, abstractmethod
from typing import Tuple

class RetrievalStrategy(ABC):
    @abstractmethod
    def generate(self, query: str, **kwargs) -> Tuple[str, float]:
        """
        يجب أن تُرجع الدالة الإجابة النصية وزمن TTFT كأرقام عشرية
        """
        pass
DEFAULT_THRESHOLD = 0.25


class ModelParameters:
    BASE_MODEL = "camembert-base"

    def __init__(self, model_name: str, threshold: float | None = DEFAULT_THRESHOLD):
        self.model_name = model_name
        self.threshold = threshold

    def __str__(self) -> str:
        return f"ModelParameters(model_name='{self.model_name}', threshold={self.threshold})"

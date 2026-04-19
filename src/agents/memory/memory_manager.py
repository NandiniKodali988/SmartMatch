class MemoryManager:
    """
    Simple in-memory storage for conversation history.
    """

    def __init__(self):
        self.history = []

    def add(self, user_input: str, grounding_output: dict):
        """
        Store a new interaction.
        """
        self.history.append({
            "user_input": user_input,
            "grounding_output": grounding_output
        })

    def get_last(self):
        """
        Get the most recent grounding output.
        """
        if not self.history:
            return None
        return self.history[-1]["grounding_output"]

    def get_all(self):
        """
        Get full conversation history.
        """
        return self.history

    def clear(self):
        """
        Reset memory.
        """
        self.history = []
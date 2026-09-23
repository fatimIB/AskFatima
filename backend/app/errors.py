class RateLimitError(Exception):
    """
    Raised when Gemini reports that the API rate or quota
    limit has been reached.
    """

    pass

class AIServiceUnavailableError(Exception):
    """Raised when the AI provider is temporarily unavailable."""
    pass

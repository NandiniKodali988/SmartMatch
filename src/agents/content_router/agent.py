class ContentRouterAgent:

    def route(self, grounding_output, user_assets=None):

        description = grounding_output.get("visual_description", "").lower()
        mood = grounding_output.get("mood", "").lower()

        if user_assets:
            return "ai_compositing"

        if any(word in description for word in [
            "my trip", "today", "this morning", "i went"
        ]):
            return "local_library"

        if any(word in mood for word in [
            "dreamlike", "fantasy", "surreal"
        ]):
            return "ai_generation"

        return "unsplash"
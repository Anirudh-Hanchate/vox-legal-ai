class LegalGrader:
    @staticmethod
    def get_score(env_total_reward: float, task_difficulty: str) -> float:
        """
        Deterministic grader logic based on accumulated reward.
        Easy: Needs classification (0.25) -> scaled to 1.0
        Medium: Needs Classify + Priority + Guidance (0.70) -> scaled
        Hard: Full pipeline (1.0)
        """
        if task_difficulty == "easy":
            return 1.0 if env_total_reward >= 0.25 else 0.0
        elif task_difficulty == "medium":
            return min(env_total_reward / 0.70, 1.0)
        else:
            return env_total_reward
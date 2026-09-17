class KnowledgeBase:
    """
    Knowledge Base for IT3012 Practical 05.

    Stores:
        - Facts
        - Rules (Horn Clauses)

    Provides:
        - Forward Chaining inference
    """

    def __init__(self):
        self.facts = set()
        self.rules = []

    # ==========================================================
    # ADD FACT
    # ==========================================================

    def tell_fact(self, fact_string):
        """
        Add a fact to the Knowledge Base.
        """
        self.facts.add(fact_string)

    # ==========================================================
    # ADD RULE
    # ==========================================================

    def tell_rule(self, premise_list, conclusion_string):
        """
        Add a rule to the Knowledge Base.

        Example:

        ["TargetVisible", "HasDust"] -> "SafeToEngage"
        """
        self.rules.append(
            (premise_list, conclusion_string)
        )

    # ==========================================================
    # CLEAR FACTS
    # ==========================================================

    def clear_facts(self):
        """
        Remove all current facts.

        Rules are NOT removed.
        """
        self.facts.clear()

    # ==========================================================
    # FORWARD CHAINING
    # ==========================================================

    def forward_chain(self):
        """
        Apply Data-Driven Forward Chaining.

        Continue applying rules until no new facts
        can be inferred.
        """

        new_facts_added = True

        while new_facts_added:

            new_facts_added = False

            for premises, conclusion in self.rules:

                # Do not infer something already known
                if conclusion not in self.facts:

                    # Modus Ponens:
                    # If ALL premises are facts,
                    # infer the conclusion.
                    if all(
                        premise in self.facts
                        for premise in premises
                    ):

                        self.facts.add(conclusion)

                        new_facts_added = True
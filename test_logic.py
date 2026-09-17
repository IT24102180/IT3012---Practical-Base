from logic_engine import KnowledgeBase


def test_forward_chaining():

    kb = KnowledgeBase()

    # ==========================================================
    # ADD DOMAIN RULES
    # ==========================================================

    kb.tell_rule(
        ["TargetVisible", "HasDust"],
        "SafeToEngage"
    )

    kb.tell_rule(
        ["SafeToEngage", "BloodseekerMissing"],
        "Retreat"
    )

    # ==========================================================
    # TEST CASE 1: SAFE ENGAGEMENT
    # ==========================================================

    kb.clear_facts()

    kb.tell_fact("TargetVisible")
    kb.tell_fact("HasDust")

    kb.forward_chain()

    assert "SafeToEngage" in kb.facts, (
        "Test 1 Failed: "
        "Should deduce SafeToEngage"
    )

    assert "Retreat" not in kb.facts, (
        "Test 1 Failed: "
        "Should NOT deduce Retreat"
    )

    print(
        "Test 1 Passed: SafeToEngage deduced correctly."
    )

    # ==========================================================
    # TEST CASE 2: UNSAFE ENGAGEMENT
    # ==========================================================

    kb.clear_facts()

    kb.tell_fact("TargetVisible")
    kb.tell_fact("HasDust")
    kb.tell_fact("BloodseekerMissing")

    kb.forward_chain()

    assert "Retreat" in kb.facts, (
        "Test 2 Failed: "
        "Should deduce Retreat"
    )

    print(
        "Test 2 Passed: Retreat deduced correctly."
    )

    print(
        "\nAll Logic Engine Test Cases Passed!"
    )


if __name__ == "__main__":
    test_forward_chaining()
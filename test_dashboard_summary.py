from mas_dashboard import RunRequest, summarize_results


def test_tamas_summary_resistance_uses_attack_denominator():
    req = RunRequest(dataset="tamas", tamas_mode="both")
    results = [
        {
            "mode": "clean",
            "task_completed": True,
            "attack_resisted": True,
        },
        {
            "mode": "attack",
            "task_completed": True,
            "attack_resisted": False,
        },
        {
            "mode": "attack",
            "task_completed": True,
            "attack_resisted": True,
        },
    ]

    summary = summarize_results(results, req)

    assert summary["clean_cases"] == 1
    assert summary["attack_cases"] == 2
    assert summary["clean_completion_rate"] == 100.0
    assert summary["attack_completion_rate"] == 100.0
    assert summary["resistance_rate"] == 50.0


if __name__ == "__main__":
    test_tamas_summary_resistance_uses_attack_denominator()
    print("dashboard summary tests passed")

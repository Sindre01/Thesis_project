from my_packages.db_service.experiment_service import run_experiment_quality_checks, run_quality_checks_for_all_experiments


def run_quality_checks():
    """test all experiment quality checks"""
    errors_found = run_quality_checks_for_all_experiments(prompt_user=False)
    assert errors_found == False
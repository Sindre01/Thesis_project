import sys
sys.path.append('../../')  # Add the path to the my_packages module
from my_packages.db_service.experiment_service import confirm_testing_rerun, confirm_validation_rerun, experiment_exists, pretty_print_experiment_collections, run_experiment_quality_checks, setup_experiment_collection
experiment_name = f"signature_coverage_10_shot"
pretty_print_experiment_collections(experiment_name, limit=21, exclude_columns=["stderr", "stdout", "code_candidate", "test_result", "error_msg"])


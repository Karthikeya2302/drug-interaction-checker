import time
import mlflow
from config import config

# Set up MLflow experiment once at import time
mlflow.set_experiment(config.MLFLOW_EXPERIMENT)


def log_interaction_check(
    user_query: str,
    drug_1: str,
    drug_2: str,
    severity: str,
    confidence: float,
    faiss_score: float,
    pubmed_count: int,
    web_count: int,
    response_time: float,
    validated: bool
):
    """
    Log every query to MLflow.
    Called after Node 5 completes.
    """
    with mlflow.start_run():
        # Log what the user asked
        mlflow.log_param("user_query", user_query[:250])
        mlflow.log_param("drug_1", drug_1)
        mlflow.log_param("drug_2", drug_2)
        mlflow.log_param("validated", validated)

        # Log what the system found
        mlflow.log_param("severity", severity)

        # Log performance metrics
        mlflow.log_metric("confidence", confidence)
        mlflow.log_metric("response_time_seconds", response_time)
        mlflow.log_metric("pubmed_results_count", pubmed_count)
        mlflow.log_metric("web_results_count", web_count)
        mlflow.log_metric("faiss_score_top1", faiss_score)


def get_average_metrics() -> dict:
    """
    Pull average metrics from MLflow for dashboard display.
    Shows in Streamlit sidebar.
    """
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT)

        if not experiment:
            return {}

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=100
        )

        if not runs:
            return {}

        # Calculate averages across all runs
        confidences = [r.data.metrics.get("confidence", 0) for r in runs]
        response_times = [r.data.metrics.get("response_time_seconds", 0) for r in runs]
        faiss_scores = [r.data.metrics.get("faiss_score_top1", 0) for r in runs]

        return {
            "total_queries": len(runs),
            "avg_confidence": sum(confidences) / len(confidences),
            "avg_response_time": sum(response_times) / len(response_times),
            "avg_faiss_score": sum(faiss_scores) / len(faiss_scores)
        }

    except Exception:
        return {}
import prefect
from prefect import task, Flow, Parameter
from prefect.tasks.prefect import create_flow_run

logger = prefect.context.get("logger")


@task
def log_completion():
    logger.info("Complete")


# The end-of-run-workflow flow is specifically for
# kicking off flows when a run completes.

# Whenever a stop document hits the Prefect-Kafka consumer,
# this flow gets kicked off and runs whatever flows are
# contained within this flow block.
with Flow("end-of-run-workflow") as flow:
    stop_doc = Parameter("stop_doc")
    uid = stop_doc["run_start"]
    validation_flow = create_flow_run(
        flow_name="general-data-validation",
        project_name="ISS",
        parameters={"beamline_acronym": "iss", "uid": uid},
    )
    processing_flow = create_flow_run(
        flow_name="processing", project_name="ISS", parameters={"ref": uid}
    )
    # We can wait for all the other flows to get started by
    # setting them as upstream_tasks.
    log_completion(upstream_tasks=[validation_flow])

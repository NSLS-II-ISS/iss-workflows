from tiled.client import from_profile

import prefect
from prefect import task, Flow, Parameter

tiled_client = from_profile("nsls2", username=None)["iss"]
tiled_client_iss = tiled_client["raw"]
tiled_client_sandbox = tiled_client["sandbox"]

# Used for Prefect logging.
logger = prefect.context.get("logger")


@task
def log_uid(ref):
    run = tiled_client_iss[ref]
    full_uid = run.start["uid"]
    logger.info(f"{full_uid = }")

    # Returns work like normal.
    return full_uid


@task
def process_run(full_uid):
    logger.info(
        f"Now we have the full uid: {full_uid}" f", we can do something with it"
    )


@task
def wait_for_all_tasks():
    logger.info("All tasks completed")


with Flow("processing") as flow:
    # We use ref because we can pass in an index, a scan_id,
    # or a uid in the UI.
    ref = Parameter("ref")
    full_uid = log_uid(ref)
    process_run_task = process_run(full_uid)
    wait_for_all_tasks(upstream_tasks=[process_run_task])

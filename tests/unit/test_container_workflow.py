from pathlib import Path

import yaml


def test_container_publish_workflow_builds_confidential_verifiers_before_image() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/container.yml").read_text())
    steps = workflow["jobs"]["build-and-push"]["steps"]

    docker_build_index = next(
        index
        for index, step in enumerate(steps)
        if str(step.get("uses", "")).startswith("docker/build-push-action@")
    )
    prior_runs = "\n".join(str(step.get("run", "")) for step in steps[:docker_build_index])

    assert "confidential-preflight" in prior_runs

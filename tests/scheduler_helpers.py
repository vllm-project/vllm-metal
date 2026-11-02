# SPDX-License-Identifier: Apache-2.0
"""Shared scheduler instrumentation for serving lifecycle tests."""


def assert_v2_model_runner(llm):
    assert llm.llm_engine.vllm_config.use_v2_model_runner, (
        "This test requires Model Runner V2; check VLLM_USE_V2_MODEL_RUNNER"
    )


def track_preemptions_and_resumptions(
    scheduler, stats, *, resumption_key="resumptions"
):
    """Count preemptions and the next scheduling of each preempted request."""
    pending_preempted_ids = set()
    schedule = scheduler.schedule

    def record_schedule(*args, **kwargs):
        result = schedule(*args, **kwargs)
        stats["preemptions"] += len(result.preempted_req_ids)
        pending_preempted_ids.update(result.preempted_req_ids)
        pending_preempted_ids.difference_update(result.finished_req_ids)
        # V2 readmits preempted requests through scheduled_new_reqs.
        resumed_req_ids = pending_preempted_ids.intersection(
            result.num_scheduled_tokens
        )
        stats[resumption_key] += len(resumed_req_ids)
        pending_preempted_ids.difference_update(resumed_req_ids)
        return result

    scheduler.schedule = record_schedule

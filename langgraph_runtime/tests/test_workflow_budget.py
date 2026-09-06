import pytest

from langgraph_runtime.workflows.state import WorkflowBudgetExceeded, check_visit_budget


def test_total_visit_budget_raises_typed_exception_with_diagnostics():
    state = {
        "agent_run_id": "run-1",
        "thread_id": "thread-1",
        "node_visit_sequence": [{"node": "a"}, {"node": "b"}],
        "loop_policy": {"max_total_visits": 2, "default_max_node_visits": 1},
    }
    with pytest.raises(WorkflowBudgetExceeded) as raised:
        check_visit_budget(state, node_id="aggregator", node_type="aggregator", visit_index=1)
    assert raised.value.as_dict() == {
        "status": "exhausted",
        "limit": 2,
        "observed": 2,
        "node_id": "aggregator",
        "node_type": "aggregator",
        "visit_index": 1,
        "run_id": "run-1",
        "thread_id": "thread-1",
    }

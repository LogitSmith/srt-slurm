from unittest.mock import patch

from srtctl.core.ip_utils import _extract_ip_address, get_node_ip


def test_extract_ip_address_ignores_slurm_status_lines():
    output = "srun: Step created for StepId=123.4\n10.109.20.203\n"

    assert _extract_ip_address(output) == "10.109.20.203"


def test_get_node_ip_filters_helper_output():
    output = "srun: Step created for StepId=123.4\n10.109.20.203\n"

    with patch("srtctl.core.ip_utils._run_bash_function", return_value=(True, output)):
        assert get_node_ip("nvl72109-T02", "123") == "10.109.20.203"


def test_get_node_ip_rejects_missing_address():
    with patch(
        "srtctl.core.ip_utils._run_bash_function",
        return_value=(True, "srun: Step created for StepId=123.4"),
    ):
        assert get_node_ip("nvl72109-T02", "123") is None

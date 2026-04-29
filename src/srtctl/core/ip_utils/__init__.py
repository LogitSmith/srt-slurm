# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
IP address resolution utilities.

This module provides:
- get_node_ip(): Get IP address for a remote SLURM node (via srun)
- get_local_ip(): Get local IP address
"""

import logging
import ipaddress
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to the bash scripts directory
SCRIPTS_DIR = Path(__file__).parent

_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def _extract_ip_address(output: str) -> str | None:
    """Extract the selected IPv4 address from helper output.

    Some Slurm/Pyxis configurations print status lines such as
    ``srun: Step created ...`` on the same stream as command output. The remote
    shell helper should print exactly one address, but filter defensively here
    so those status lines cannot become part of a host/address value.
    """
    candidates: list[str] = []
    for match in _IPV4_RE.finditer(output):
        value = match.group(0)
        try:
            ip = ipaddress.ip_address(value)
        except ValueError:
            continue
        if ip.is_unspecified or ip.is_loopback or ip.is_link_local:
            continue
        candidates.append(str(ip))
    return candidates[-1] if candidates else None


def _run_bash_function(
    script: str,
    function: str,
    args: list[str],
    timeout: float = 30.0,
) -> tuple[bool, str]:
    """Run a bash function from a script file.

    Args:
        script: Name of the script file
        function: Name of the function to call
        args: List of arguments to pass to the function
        timeout: Command timeout in seconds

    Returns:
        Tuple of (success: bool, output: str)
    """
    script_path = SCRIPTS_DIR / script
    if not script_path.exists():
        logger.error("Script not found: %s", script_path)
        return False, f"Script not found: {script_path}"

    # Build bash command that sources the script and calls the function
    quoted_args = " ".join(f'"{arg}"' for arg in args)
    bash_cmd = f'source "{script_path}" && {function} {quoted_args}'

    try:
        result = subprocess.run(
            ["bash", "-c", bash_cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            logger.debug(
                "Bash function %s failed (exit %d): %s",
                function,
                result.returncode,
                result.stderr or result.stdout,
            )
            return False, result.stderr or result.stdout

    except subprocess.TimeoutExpired:
        logger.error("Timeout running bash function %s", function)
        return False, "Timeout"
    except Exception as e:
        logger.error("Error running bash function %s: %s", function, e)
        return False, str(e)


def get_node_ip(
    node: str,
    slurm_job_id: str | None = None,
    network_interface: str | None = None,
    timeout: float = 30.0,
) -> str | None:
    """Get IP address for a remote SLURM node.

    Uses srun to execute IP resolution on the target node.

    Args:
        node: Hostname of the node to query
        slurm_job_id: SLURM job ID for srun context
        network_interface: Specific network interface to use
        timeout: Command timeout in seconds

    Returns:
        IP address string, or None if resolution failed
    """
    success, output = _run_bash_function(
        "get_node_ip.sh",
        "get_node_ip",
        [node, slurm_job_id or "", network_interface or ""],
        timeout=timeout,
    )

    if success and output:
        ip = _extract_ip_address(output)
        if ip:
            if ip != output:
                logger.debug("Filtered IP helper output for %s: %r -> %s", node, output, ip)
            else:
                logger.debug("Resolved IP for %s: %s", node, ip)
            return ip
        logger.error("IP helper for node %s did not return a usable address: %s", node, output)
        return None
    else:
        logger.error("Failed to get IP for node %s: %s", node, output)
        return None


def get_local_ip(network_interface: str | None = None) -> str:
    """Get local IP address.

    Tries multiple methods:
    1. Specific network interface (if provided)
    2. hostname -I (gets first non-loopback IP)
    3. ip route get 8.8.8.8 (finds default source IP)

    Args:
        network_interface: Specific network interface to use

    Returns:
        IP address string, or "127.0.0.1" if all methods fail
    """
    success, output = _run_bash_function(
        "get_node_ip.sh",
        "get_local_ip",
        [network_interface or ""],
        timeout=5.0,
    )

    if success and output:
        return output
    else:
        logger.warning("Could not determine local IP, using 127.0.0.1")
        return "127.0.0.1"

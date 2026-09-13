"""Bounded, opt-in, read-only diagnostics. Python standard library only."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import http.client
import json
from pathlib import Path
import math
import os
import re
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import Literal, TypedDict
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


MAX_BYTES = 1_048_576
TIMEOUT = 3.0


class Field(TypedDict):
    value: str | float | int | bool | list[int] | None
    status: Literal["available", "unavailable", "not_measured"]
    reason: str
    unit: str


def field(value: str | float | int | bool | list[int] | None = None,
          reason: str = "unsupported_or_missing", unit: str = "") -> Field:
    return {"value": value, "status": "available" if value is not None else "unavailable",
            "reason": "observed" if value is not None else reason, "unit": unit}


def number(value: object, unit: str = "", reason: str = "unsupported_or_missing") -> Field:
    if isinstance(value, bool):
        return field(reason=reason, unit=unit)
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(?:\s*(?:MiB|W|MHz|C|%|x|us|ms|kB))?", str(value).strip())
    if match:
        parsed = float(match[1])
        if math.isfinite(parsed) and parsed <= 1e20:
            return field(parsed, unit=unit)
    return field(reason=reason, unit=unit)


def version(value: object) -> Field:
    if isinstance(value, str) and re.fullmatch(r"[0-9]{1,4}(?:\.[0-9]{1,4}){1,3}(?:(?:a|b|rc|\.post)[0-9]{1,4})?(?:\+cu[0-9]{2,4})?", value):
        return field(value)
    return field(reason="unsupported_or_redacted")


def gpu_model(value: object) -> Field:
    pattern = (r"(?:NVIDIA )?(?:GeForce RTX [2345][0-9]{3}(?: Ti)?(?: SUPER)?|"
               r"RTX PRO [2456]000 Blackwell (?:Workstation Edition|Server Edition|Max-Q Workstation Edition)|"
               r"RTX (?:A[2456]000|[2456]000 Ada Generation)|"
               r"[ABHLV][0-9]{2,3}(?:-[A-Z0-9]{2,6})?(?: [0-9]{1,3}GB)?(?: PCIe| SXM[0-9]?| NVL)?)")
    if isinstance(value, str) and re.fullmatch(pattern, value):
        return field(value)
    return field(reason="unsupported_or_redacted")


GPU_FIELDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "vram_mib": ("MiB", ("fb_memory_usage/total",)),
    "pcie_current_generation": ("generation", ("pci/pci_gpu_link_info/pcie_gen/current_link_gen",)),
    "pcie_max_generation": ("generation", ("pci/pci_gpu_link_info/pcie_gen/max_link_gen",)),
    "pcie_current_width": ("lanes", ("pci/pci_gpu_link_info/link_widths/current_link_width",)),
    "pcie_max_width": ("lanes", ("pci/pci_gpu_link_info/link_widths/max_link_width",)),
    "power_w": ("W", ("gpu_power_readings/power_draw", "power_readings/power_draw", "gpu_power_readings/instant_power_draw")),
    "power_limit_w": ("W", ("gpu_power_readings/power_limit", "power_readings/power_limit", "gpu_power_readings/requested_power_limit")),
    "default_power_limit_w": ("W", ("gpu_power_readings/default_power_limit", "power_readings/default_power_limit")),
    "max_power_limit_w": ("W", ("gpu_power_readings/max_power_limit", "power_readings/max_power_limit")),
    "sm_clock_mhz": ("MHz", ("clocks/sm_clock",)),
    "memory_clock_mhz": ("MHz", ("clocks/mem_clock",)),
    "temperature_c": ("C", ("temperature/gpu_temp",)),
    "gpu_util_percent": ("percent", ("utilization/gpu_util",)),
    "memory_util_percent": ("percent", ("utilization/memory_util",)),
}
EVENTS = ("gpu_idle", "applications_clocks_setting", "sw_power_cap", "hw_slowdown",
          "hw_thermal_slowdown", "hw_power_brake_slowdown", "sync_boost", "sw_thermal_slowdown")


def parse_gpu(raw: Raw) -> dict[str, object]:
    reason = raw.reason if raw.reason != "available" else "unsupported_or_missing"
    root = ET.Element("missing")
    try:
        if "<!ENTITY" in raw.text or len(raw.text) > MAX_BYTES:
            reason = "invalid_or_oversized"
        elif raw.text:
            root = ET.fromstring(raw.text)
    except (ET.ParseError, RecursionError):
        reason = "invalid_format"
    gpu = root.find("gpu")
    if gpu is None:
        gpu = ET.Element("missing")
    result: dict[str, object] = {}
    for key, (unit, paths) in GPU_FIELDS.items():
        values = [gpu.findtext(path) for path in paths]
        result[key] = number(next((value for value in values if value is not None), None), unit, reason)
    result["model"] = gpu_model(gpu.findtext("product_name")) if raw.text else field(reason=reason)
    result["driver_version"] = version(root.findtext("driver_version"))
    result["driver_cuda_supported_version"] = version(root.findtext("cuda_version"))
    pstate = gpu.findtext("performance_state", "")
    result["pstate"] = field(pstate) if re.fullmatch(r"P[0-9]{1,2}", pstate) else field(reason=reason)
    events: dict[str, Field] = {}
    counters: dict[str, Field] = {}
    for event in EVENTS:
        value = None
        for prefix in ("clocks_event_reasons", "clocks_throttle_reasons"):
            for tag in (prefix, prefix.removesuffix("s")):
                text = gpu.findtext(f"{prefix}/{tag}_{event}")
                if text in ("Active", "Not Active"):
                    value = text == "Active"
        events[event] = field(value, reason=reason)
        counter_tag = {"sw_thermal_slowdown": "sw_therm_slowdown",
                       "hw_thermal_slowdown": "hw_therm_slowdown",
                       "hw_power_brake_slowdown": "hw_power_brake"}.get(event, event)
        counter_text = gpu.findtext(f"clocks_event_reasons_counters/{event}")
        if counter_text is None:
            counter_text = gpu.findtext(f"clocks_event_reasons_counters/clocks_event_reasons_counters_{counter_tag}")
        counters[event] = number(counter_text, "us", reason)
    result["clock_events"] = events
    result["event_counters_us"] = counters
    return result


def parse_matrix(raw: Raw, selected: list[int], capability: bool = False) -> dict[str, dict[str, Field]]:
    header: list[str] = []
    rows: dict[str, list[str]] = {}
    for line in re.sub(r"\x1b\[[0-9;]*m", "", raw.text).splitlines():
        words = line.split()
        if not header and words and re.fullmatch(r"GPU[0-9]+", words[0]):
            header = []
            for word in words:
                if not re.fullmatch(r"GPU[0-9]+", word):
                    break
                header.append(word)
        elif words and re.fullmatch(r"GPU[0-9]+", words[0]):
            rows[words[0]] = words[1:len(header) + 1]
    result: dict[str, dict[str, Field]] = {}
    for row_index, local_row in enumerate(selected):
        row: dict[str, Field] = {}
        for col_index, local_col in enumerate(selected):
            token = ""
            if f"GPU{local_col}" in header:
                offset = header.index(f"GPU{local_col}")
                values = rows.get(f"GPU{local_row}", [])
                if offset < len(values):
                    token = values[offset]
            reason = raw.reason if raw.reason != "available" else "unsupported_or_missing"
            if capability:
                entry = field(True) if token == "OK" else field(False) if token in ("NS", "CNS", "GNS", "TNS") else field(reason=reason)
            else:
                entry = field(token) if re.fullmatch(r"X|SYS|NODE|PHB|PXB|PIX|NV[0-9]{1,2}", token) else field(reason=reason)
            row[f"GPU{col_index}"] = entry
        result[f"GPU{row_index}"] = row
    return result


def collect_gpus(selected: list[int], include_topology: bool = True) -> dict[str, object]:
    devices: dict[str, object] = {}
    for synthetic, index in enumerate(selected):
        devices[f"GPU{synthetic}"] = parse_gpu(run_bounded(["nvidia-smi", "-q", "-x", "-i", str(index)]))
    if any(device["driver_cuda_supported_version"]["value"] is None for device in devices.values()):
        header = run_bounded(["nvidia-smi", "-i", ",".join(map(str, selected))])
        match = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", header.text)
        for device in devices.values():
            if device["driver_cuda_supported_version"]["value"] is None:
                device["driver_cuda_supported_version"] = version(match[1]) if match else field(reason=header.reason if header.reason != "available" else "unsupported_or_missing")
    result: dict[str, object] = {"devices": devices}
    if include_topology:
        result["topology"] = parse_matrix(run_bounded(["nvidia-smi", "topo", "-m"]), selected)
        for kind, flag in (("read", "r"), ("write", "w")):
            result[f"p2p_{kind}_capability"] = parse_matrix(run_bounded(["nvidia-smi", "topo", "-p2p", flag]), selected, True)
    return result


def parse_cpu_list(text: str) -> Field:
    if len(text) > 65536 or not re.fullmatch(r"[0-9,-]+\s*", text):
        return field(reason="invalid_or_unavailable")
    values: set[int] = set()
    for part in text.strip().split(","):
        bounds = part.split("-")
        if len(bounds) > 2 or not all(bound.isdigit() and len(bound) <= 5 for bound in bounds):
            return field(reason="invalid_or_unavailable")
        first, last = int(bounds[0]), int(bounds[-1])
        if last < first or last > 65535 or len(values) + last - first + 1 > 65536:
            return field(reason="invalid_or_unavailable")
        values.update(range(first, last + 1))
    return field(sorted(values))


def cpu_model(value: object) -> Field:
    pattern = (r"AMD (?:Ryzen (?:[3579] [0-9]{4,5}[A-Z0-9]{0,4}|Threadripper(?: PRO)? [0-9]{4,5}[A-Z]{0,3})"
               r"(?: [0-9]{1,3}-Core(?:s| Processor)?)?|EPYC [0-9]{4}[A-Z0-9]{0,3}(?: [0-9]{1,3}-Core Processor)?)|"
               r"Intel(?:\(R\))? (?:Core(?:\(TM\))? (?:i[3579]-[0-9]{4,5}[A-Z]{0,3}|Ultra [579] [0-9]{3}[A-Z]{0,2})|"
               r"Xeon(?:\(R\))?(?: (?:CPU|Gold|Silver|Platinum|Bronze))? [A-Z0-9-]{3,12})(?: CPU)?(?: @ [0-9]\.[0-9]{1,2}GHz)?|"
               r"Apple M[1-9](?: Pro| Max| Ultra)?")
    return field(value) if isinstance(value, str) and re.fullmatch(pattern, value) else field(reason="unsupported_or_redacted")


def pairs(raw: Raw) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in raw.text.splitlines():
        words = line.replace(":", " ", 1).split()
        if len(words) >= 2:
            result[words[0]] = words[1]
    return result


def selected_numbers(raw: Raw, names: dict[str, str], unit: str) -> dict[str, Field]:
    source = pairs(raw)
    return {output: number(source.get(key), unit, raw.reason if raw.reason != "available" else "unsupported_or_missing")
            for output, key in names.items()}


def parse_pressure(raw: Raw) -> dict[str, Field]:
    result: dict[str, Field] = {}
    rows: dict[str, dict[str, str]] = {}
    for line in raw.text.splitlines():
        words = line.split()
        if words and words[0] in ("some", "full"):
            rows[words[0]] = dict(word.split("=", 1) for word in words[1:] if "=" in word)
    for kind in ("some", "full"):
        for key in ("avg10", "avg60", "avg300", "total"):
            result[f"{kind}_{key}" + ("_us" if key == "total" else "")] = number(
                rows.get(kind, {}).get(key), "us" if key == "total" else "percent",
                raw.reason if raw.reason != "available" else "unsupported_or_missing")
    return result


def collect_host(proc_root: Path = Path("/proc"), sys_root: Path = Path("/sys")) -> dict[str, object]:
    cpu = read_bounded(proc_root / "cpuinfo")
    model = next((line.partition(":")[2].strip() for line in cpu.text.splitlines()
                  if line.partition(":")[0].strip() == "model name"), None)
    status = pairs(read_bounded(proc_root / "self/status"))
    memory = selected_numbers(read_bounded(proc_root / "meminfo"), {
        "total": "MemTotal", "available": "MemAvailable", "swap_total": "SwapTotal", "swap_free": "SwapFree"}, "KiB")
    vm = selected_numbers(read_bounded(proc_root / "vmstat"), {
        "swap_in_pages": "pswpin", "swap_out_pages": "pswpout", "page_in_kib": "pgpgin",
        "page_out_kib": "pgpgout", "oom_kills": "oom_kill"}, "counter_since_boot")
    stat = read_bounded(proc_root / "stat")
    cpu_line = next((line.split()[1:9] for line in stat.text.splitlines() if line.startswith("cpu ")), [])
    cpu_ticks = {key: number(cpu_line[index] if index < len(cpu_line) else None, "ticks_since_boot")
                 for index, key in enumerate(("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal"))}
    node_root = sys_root / "devices/system/node"
    online = parse_cpu_list(read_bounded(node_root / "online").text)
    numa: dict[str, object] = {}
    if online["value"] is not None and len(online["value"]) <= 256:
        for index in online["value"]:
            node = node_root / f"node{index}"
            distances = read_bounded(node / "distance").text.split()
            valid = len(distances) <= 256 and bool(distances) and all(re.fullmatch(r"[0-9]{1,5}", value) for value in distances)
            numa[f"node{index}"] = {"cpus": parse_cpu_list(read_bounded(node / "cpulist").text),
                                    "distances": field([int(value) for value in distances]) if valid else field()}
    return {"cpu_model": cpu_model(model) if cpu.reason == "available" else field(reason=cpu.reason),
            "logical_cpu_count": field(os.cpu_count()), "cpuset": parse_cpu_list(status.get("Cpus_allowed_list", "")),
            "memory_nodes": parse_cpu_list(status.get("Mems_allowed_list", "")),
            "numa_online": online, "numa": numa, "memory_kib": memory, "vm_counters": vm,
            "cpu_ticks": cpu_ticks, "pressure": {resource: parse_pressure(read_bounded(proc_root / "pressure" / resource))
                                                  for resource in ("cpu", "memory", "io")}}


def cgroup_locations(proc_root: Path, subject: str) -> tuple[int | None, dict[str, tuple[Path, Path]]]:
    membership = read_bounded(proc_root / subject / "cgroup")
    mounts = read_bounded(proc_root / "self/mountinfo")
    groups: dict[str, str] = {}
    for line in membership.text.splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[2].startswith("/") and ".." not in Path(parts[2]).parts:
            for controller in parts[1].split(","):
                groups[controller] = parts[2]
    locations: dict[str, tuple[Path, Path]] = {}
    found_version = None
    for line in mounts.text.splitlines():
        left, separator, right = line.partition(" - ")
        before, after = left.split(), right.split()
        if not separator or len(before) < 6 or len(after) < 3 or after[0] not in ("cgroup", "cgroup2"):
            continue
        mount_root, mount_point = before[3], before[4]
        if "\\" in mount_root or "\\" in mount_point:
            continue
        controllers = ("",) if after[0] == "cgroup2" else tuple(after[2].split(","))
        for controller in controllers:
            if controller not in groups:
                continue
            try:
                relative = Path(groups[controller]).relative_to(mount_root)
            except ValueError:
                continue
            if not Path(mount_point).is_absolute() or ".." in Path(mount_point).parts:
                continue
            location = (Path(mount_point) / relative, Path(mount_point))
            if controller == "":
                for name in ("cpu", "memory", "cpuset"):
                    locations[name] = location
                found_version = 2
            elif controller in ("cpu", "memory", "cpuset"):
                locations[controller] = location
                found_version = 1
    return found_version, locations


def hierarchy_limit(location: tuple[Path, Path] | None, kind: str, v2: bool) -> Field:
    if location is None:
        return field(reason="cgroup_unavailable")
    current, mount = location
    observed: list[float] = []
    unknown = False
    for _ in range(32):
        if kind == "cpu":
            if v2:
                raw = read_bounded(current / "cpu.max")
                if current == mount and raw.reason == "not_found":
                    break
                parts = raw.text.split()
            else:
                parts = [read_bounded(current / "cpu.cfs_quota_us").text.strip(),
                         read_bounded(current / "cpu.cfs_period_us").text.strip()]
            if len(parts) == 2 and parts[0] in ("max", "-1"):
                pass
            elif len(parts) == 2:
                quota, period = number(parts[0])["value"], number(parts[1])["value"]
                if quota is not None and period is not None and period > 0:
                    observed.append(quota / period)
                else:
                    unknown = True
            else:
                unknown = True
        else:
            filename = "memory.swap.max" if kind == "swap" else "memory.max" if v2 else "memory.limit_in_bytes"
            raw = read_bounded(current / filename)
            if v2 and current == mount and raw.reason == "not_found":
                break
            text = raw.text.strip()
            value = number(text)["value"]
            if text == "max" or (not v2 and value is not None and value >= 2**60):
                pass
            elif value is not None:
                observed.append(value)
            else:
                unknown = True
        if current == mount:
            break
        current = current.parent
    else:
        return field(reason="hierarchy_depth_limit")
    if unknown:
        return field(reason="hierarchy_partially_unavailable")
    return field(min(observed) if observed else "unlimited", unit="cores" if kind == "cpu" else "bytes")


def collect_cgroup(proc_root: Path = Path("/proc"), subject: str = "self") -> dict[str, object]:
    detected, locations = cgroup_locations(proc_root, subject)
    v2 = detected == 2

    def read(controller: str, filename: str) -> Raw:
        location = locations.get(controller)
        return read_bounded(location[0] / filename) if location else Raw(reason="cgroup_unavailable")

    counters = selected_numbers(read("cpu", "cpu.stat"), {
        "periods": "nr_periods", "throttled_periods": "nr_throttled",
        "throttled_us": "throttled_usec" if v2 else "throttled_time", "usage_us": "usage_usec"}, "counter_since_cgroup_creation")
    if not v2 and counters["throttled_us"]["value"] is not None:
        counters["throttled_us"]["value"] /= 1000
    if v2:
        events = selected_numbers(read("memory", "memory.events"), {key: key for key in ("low", "high", "max", "oom", "oom_kill")}, "count")
    else:
        events = {"limit_failures": number(read("memory", "memory.failcnt").text.strip(), "count")}
    return {"version": field(detected, reason="cgroup_unavailable"),
            "cpu_limit_cores": hierarchy_limit(locations.get("cpu"), "cpu", v2),
            "memory_limit_bytes": hierarchy_limit(locations.get("memory"), "memory", v2),
            "swap_limit_bytes": hierarchy_limit(locations.get("memory"), "swap", True) if v2 else field(reason="v1_combined_memsw_not_swap_only"),
            "memory_current_bytes": number(read("memory", "memory.current" if v2 else "memory.usage_in_bytes").text.strip(), "bytes"),
            "cpuset": parse_cpu_list(read("cpuset", "cpuset.cpus.effective" if v2 else "cpuset.cpus").text),
            "memory_nodes": parse_cpu_list(read("cpuset", "cpuset.mems.effective" if v2 else "cpuset.mems").text),
            "cpu_counters": counters, "memory_events": events}


SETTING_NUMBERS = {
    "tp_size": (1, 1024), "speculative_num_steps": (0, 128), "speculative_eagle_topk": (1, 128),
    "speculative_num_draft_tokens": (1, 1024), "context_length": (1, 16_777_216),
    "max_total_tokens": (1, 100_000_000), "max_running_requests": (1, 1_000_000),
    "chunked_prefill_size": (-1, 16_777_216), "max_prefill_tokens": (1, 16_777_216),
    "cuda_graph_max_bs": (1, 65536), "cuda_graph_max_bs_decode": (1, 65536),
    "max_mamba_cache_size": (0, 100_000_000), "mamba_track_interval": (1, 1_000_000),
    "mem_fraction_static": (0, 1), "page_size": (1, 65536),
}
BACKENDS = ("auto", "triton", "flashinfer", "flashinfer_cutlass", "flashinfer_trtllm", "fa3", "fa4", "torch_native", "cutlass", "flashattn")
SETTING_ENUMS = {
    "quantization": ("modelopt_mixed", "modelopt_fp4", "nvfp4", "fp8", "awq", "gptq", "none", "mxfp4"),
    "speculative_draft_model_quantization": ("modelopt_mixed", "modelopt_fp4", "nvfp4", "fp8", "none"),
    "speculative_algorithm": ("NEXTN", "EAGLE", "EAGLE3", "STANDALONE", "NGRAM", "none"),
    "attention_backend": BACKENDS, "moe_runner_backend": BACKENDS, "speculative_moe_runner_backend": BACKENDS,
    "kv_cache_dtype": ("auto", "bfloat16", "float16", "fp8_e4m3", "fp8_e5m2"),
    "mamba_ssm_dtype": ("float32", "bfloat16", "float16"),
    "mamba_radix_cache_strategy": ("no_buffer", "extra_buffer"),
}
SETTING_BOOLS = ("ple_offload_embedding", "disable_cuda_graph", "disable_prefill_cuda_graph",
                 "disable_overlap_schedule", "disable_radix_cache", "enable_mixed_chunk",
                 "enable_dp_attention", "enable_dp_lm_head")
ENV_BOOLS = ("SGLANG_PRIVATE_DRAFT_NVFP4_A16", "SGLANG_PLE_PACKED_NVFP4", "SGLANG_PLE_PACKED_FP8_REFERENCE",
             "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE", "NCCL_IB_DISABLE", "NCCL_CUMEM_ENABLE")
ENV_ENUMS = {"NCCL_P2P_LEVEL": ("LOC", "NVL", "PIX", "PXB", "PHB", "SYS"),
             "NCCL_ALGO": ("Ring", "Tree", "CollnetDirect", "CollnetChain", "NVLS", "NVLSTree"),
             "NCCL_PROTO": ("Simple", "LL", "LL128")}


def safe_json(raw: Raw) -> object:
    if raw.reason != "available" or len(raw.text) > MAX_BYTES:
        return None
    try:
        return json.loads(raw.text)
    except (ValueError, RecursionError):
        return None


def mapping(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def boolean(value: object) -> Field:
    if type(value) is bool:
        return field(value)
    if type(value) in (str, int) and value in ("0", "1", "true", "false", 0, 1):
        return field(value in ("1", "true", 1))
    return field(reason="unsupported_or_redacted")


def enum_field(value: object, allowed: tuple[str, ...]) -> Field:
    return field(value) if type(value) is str and value in allowed else field(reason="unsupported_or_redacted")


def parse_settings(source: object) -> dict[str, Field]:
    source = mapping(source)
    result: dict[str, Field] = {}
    for key, (lower, upper) in SETTING_NUMBERS.items():
        value = source.get(key)
        parsed = None
        if type(value) in (str, int, float) and re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", str(value)) and len(str(value)) < 32:
            candidate = float(value)
            if lower <= candidate <= upper and (key == "mem_fraction_static" or candidate.is_integer()):
                parsed = candidate
        result[key] = field(parsed, reason="unsupported_or_redacted")
    for key, allowed in SETTING_ENUMS.items():
        result[key] = enum_field(source.get(key), allowed)
    for key in SETTING_BOOLS:
        result[key] = boolean(source.get(key))
    return result


def settings_from_args(args: object) -> dict[str, Field]:
    selected: dict[str, object] = {}
    duplicates: set[str] = set()
    if not isinstance(args, list) or len(args) > 8192:
        return parse_settings({})
    for index, arg in enumerate(args):
        if not isinstance(arg, str) or not arg.startswith("--"):
            continue
        key, equal, value = arg[2:].partition("=")
        key = key.replace("-", "_")
        key = {"tp": "tp_size", "tensor_parallel_size": "tp_size"}.get(key, key)
        if key not in SETTING_NUMBERS and key not in SETTING_ENUMS and key not in SETTING_BOOLS:
            continue
        if not equal:
            value = True if key in SETTING_BOOLS else args[index + 1] if index + 1 < len(args) else None
        if key in selected:
            duplicates.add(key)
        selected[key] = value
    for key in duplicates:
        selected[key] = None
    return parse_settings(selected)


def parse_container(source: object) -> dict[str, object]:
    source = mapping(source)
    config = mapping(source.get("Config"))
    environment: dict[str, object] = {}
    duplicates: set[str] = set()
    raw_env = config.get("Env")
    if isinstance(raw_env, list) and len(raw_env) <= 8192:
        for item in raw_env:
            if isinstance(item, str):
                key, separator, value = item.partition("=")
                if separator and (key in ENV_BOOLS or key in ENV_ENUMS):
                    if key in environment:
                        duplicates.add(key)
                    environment[key] = value
    for key in duplicates:
        environment[key] = None
    safe_env = {key: boolean(environment.get(key)) for key in ENV_BOOLS}
    safe_env.update({key: enum_field(environment.get(key), allowed) for key, allowed in ENV_ENUMS.items()})
    digest = source.get("Image")
    return {"image_digest": field(digest) if isinstance(digest, str) and re.fullmatch(r"sha256:[a-f0-9]{64}", digest) else field(reason="digest_unavailable"),
            "settings": settings_from_args(source.get("Args")), "environment": safe_env,
            "shm_limit_bytes": number(mapping(source.get("HostConfig")).get("ShmSize"), "bytes")}


VERSION_PROBE = (
    "import json,site; from importlib import metadata; "
    "wanted={'sglang':'sglang','torch':'torch','nvidia-cuda-runtime-cu12':'cuda_runtime_package',"
    "'nvidia-cuda-runtime-cu13':'cuda_runtime_package'}; "
    "result={wanted[d.metadata['Name']]:d.version for d in metadata.distributions(path=site.getsitepackages()) "
    "if d.metadata['Name'] in wanted}; print(json.dumps(result))"
)


DOCKER_SOCKET = "/var/run/docker.sock"


def docker_pid_namespace_verified() -> bool:
    try:
        if any(Path(marker).exists() for marker in ("/.dockerenv", "/run/.containerenv", "/run/systemd/container")):
            return False
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(TIMEOUT)
            connection.connect(DOCKER_SOCKET)
            peer_pid, _, _ = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
        if peer_pid <= 0:
            return False
        return all(os.readlink(f"/proc/self/ns/{namespace}") == os.readlink(f"/proc/{peer_pid}/ns/{namespace}")
                   for namespace in ("pid", "mnt"))
    except (OSError, ValueError, AttributeError):
        return False


def collect_container(container: str, context: dict[str, str] | None = None) -> dict[str, object]:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("DOCKER_")}
    versions: dict = {}
    with tempfile.TemporaryDirectory(prefix="diagnostic-docker-") as config:
        command = ["docker", "--host", "unix://" + DOCKER_SOCKET, "--config", config]
        raw = run_bounded([*command, "inspect", "--type=container", "--", container], env=environment)
        decoded = safe_json(raw)
        source = mapping(decoded[0]) if isinstance(decoded, list) and len(decoded) == 1 else {}
        if source:
            versions = mapping(safe_json(run_bounded([*command, "exec", container, "python3", "-I", "-S", "-B", "-c", VERSION_PROBE], env=environment)))
    result = parse_container(source)
    result["inspect_status"] = field(True) if source else field(reason=raw.reason if raw.reason != "available" else "invalid_format")
    result["runtime_packages"] = {key: version(versions.get(key)) for key in ("sglang", "torch", "cuda_runtime_package")}
    pid = mapping(source.get("State")).get("Pid")
    if context is not None:
        context.pop("subject", None)
    result["cgroup"] = {"status": field(reason="container_pid_unavailable")}
    if type(pid) is int and 0 < pid < 2**31:
        if docker_pid_namespace_verified():
            if context is not None:
                context["subject"] = str(pid)
            result["cgroup"] = collect_cgroup(subject=str(pid))
        else:
            result["cgroup"] = {"status": field(reason="container_pid_namespace_unverified")}
    return result


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def validate_endpoint(endpoint: str) -> str:
    if len(endpoint) > 2048 or re.search(r"[\s\x00-\x1f\x7f]", endpoint):
        raise ValueError("invalid endpoint")
    try:
        parts = urllib.parse.urlsplit(endpoint)
        valid = (parts.scheme in ("http", "https") and parts.hostname and parts.username is None
                 and parts.password is None and parts.path in ("", "/") and not parts.query and not parts.fragment
                 and (parts.port is None or 1 <= parts.port <= 65535))
        if not valid or not re.fullmatch(r"[A-Za-z0-9.:[\]-]+", parts.netloc):
            raise ValueError("invalid endpoint")
    except ValueError:
        raise ValueError("invalid endpoint") from None
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def fetch_endpoint(endpoint: str, route: str, token: str | None = None,
                   timeout: float = TIMEOUT, limit: int = MAX_BYTES) -> Raw:
    deadline = time.monotonic() + timeout
    base = validate_endpoint(endpoint)
    if route not in ("/get_server_info", "/metrics"):
        raise ValueError("unsupported route")
    if token is not None and (len(token) > 8192 or re.search(r"[^\x21-\x7e]", token)):
        return Raw(reason="invalid_token")
    environment = dict(os.environ)
    environment.pop("DIAGNOSTIC_HTTP_TOKEN", None)
    if token:
        environment["DIAGNOSTIC_HTTP_TOKEN"] = token
    payload = json.dumps({"endpoint": base, "route": route, "timeout": timeout, "limit": limit}).encode("utf-8")
    try:
        with subprocess.Popen([sys.executable, "-I", "-S", "-B", str(Path(__file__).resolve()), "--http-worker"],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              env=environment, shell=False) as process:
            try:
                output, _ = process.communicate(payload, timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                return Raw(reason="timeout")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
            if time.monotonic() >= deadline:
                return Raw(reason="timeout")
            if process.returncode or len(output) > 6 * limit + 256:
                return Raw(reason="connection_failed")
        result = json.loads(output)
        if not isinstance(result, dict) or set(result) != {"text", "reason"}:
            return Raw(reason="connection_failed")
        if result["reason"] == "available" and isinstance(result["text"], str) and len(result["text"]) <= limit:
            return Raw(result["text"])
        if result["reason"] in ("invalid_token", "redirect_refused", "http_error", "timeout", "oversized", "connection_failed"):
            return Raw(reason=result["reason"])
    except (OSError, ValueError, RecursionError):
        pass
    return Raw(reason="connection_failed")


def fetch_endpoint_worker(endpoint: str, route: str, token: str | None = None,
                          timeout: float = TIMEOUT, limit: int = MAX_BYTES) -> Raw:
    base = validate_endpoint(endpoint)
    if route not in ("/get_server_info", "/metrics"):
        raise ValueError("unsupported route")
    if token is not None and (len(token) > 8192 or re.search(r"[^\x21-\x7e]", token)):
        return Raw(reason="invalid_token")
    headers = {"Accept": "application/json" if route == "/get_server_info" else "text/plain"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(base + route, headers=headers, method="GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    deadline = time.monotonic() + timeout
    try:
        with opener.open(request, timeout=timeout) as response:
            if 300 <= response.status < 400:
                return Raw(reason="redirect_refused")
            if response.status != 200:
                return Raw(reason="http_error")
            chunks: list[bytes] = []
            size = 0
            while size <= limit:
                if time.monotonic() >= deadline:
                    return Raw(reason="timeout")
                chunk = response.read1(min(8192, limit + 1 - size))
                if not chunk:
                    return Raw(b"".join(chunks).decode("utf-8", errors="replace"))
                chunks.append(chunk)
                size += len(chunk)
            return Raw(reason="oversized")
    except urllib.error.HTTPError as error:
        reason = "redirect_refused" if 300 <= error.code < 400 else "http_error"
        error.close()
        return Raw(reason=reason)
    except TimeoutError:
        return Raw(reason="timeout")
    except (OSError, ValueError, http.client.HTTPException):
        return Raw(reason="connection_failed")


def http_worker_main() -> int:
    try:
        request = json.loads(sys.stdin.buffer.read(16384))
        result = fetch_endpoint_worker(request["endpoint"], request["route"],
                                       os.environ.get("DIAGNOSTIC_HTTP_TOKEN"), request["timeout"], request["limit"])
    except Exception:
        result = Raw(reason="connection_failed")
    sys.stdout.write(json.dumps({"text": result.text, "reason": result.reason}))
    return 0


METRICS: dict[str, tuple[str, str]] = {
    "running_requests": ("sglang:num_running_reqs", "requests"),
    "queued_requests": ("sglang:num_queue_reqs", "requests"),
    "generation_throughput": ("sglang:gen_throughput", "tokens/s_reported"),
    "accept_length": ("sglang:spec_accept_length", "tokens/request_verification_reported"),
    "accept_rate": ("sglang:spec_accept_rate", "ratio"),
    "cache_hit_rate": ("sglang:cache_hit_rate", "ratio"),
    "token_usage": ("sglang:token_usage", "ratio"),
    "mamba_usage": ("sglang:mamba_usage", "ratio"),
    "mamba_used_tokens": ("sglang:mamba_used_tokens", "tokens"),
    "mamba_available_tokens": ("sglang:mamba_available_tokens", "tokens"),
    "mamba_evictable_tokens": ("sglang:mamba_evictable_tokens", "tokens"),
    "generation_tokens": ("sglang:generation_tokens_total", "tokens_counter_completed_requests"),
    "request_verifications": ("sglang:spec_verify_calls_total", "request_verifications_counter_completed_requests"),
    "cached_tokens": ("sglang:cached_tokens_total", "tokens_counter"),
}


def parse_metrics(raw: Raw, identities: dict[str, str] | None = None) -> dict[str, Field]:
    wanted = {name for name, _ in METRICS.values()}
    observed: dict[str, list[tuple[str, Field]]] = {}
    for line in raw.text.splitlines():
        if len(line) > 16384 or line.startswith("#"):
            continue
        match = re.fullmatch(r'([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+[0-9]+)?', line)
        if match and match[1] in wanted:
            value = None
            try:
                parsed = float(match[3])
                if math.isfinite(parsed) and 0 <= parsed <= 1e20:
                    value = parsed
            except ValueError:
                pass
            observed.setdefault(match[1], []).append((match[2] or "", field(value, reason="invalid_metric")))
    result: dict[str, Field] = {}
    for key, (name, unit) in METRICS.items():
        entries = observed.get(name, [])
        if len(entries) == 1:
            identity, value = entries[0]
            if unit == "ratio" and value["value"] is not None and value["value"] > 1:
                value = field(reason="invalid_metric")
            result[key] = {**value, "unit": unit}
            if identities is not None:
                identities[key] = identity
        else:
            result[key] = field(reason="multiple_series_ambiguous" if entries else raw.reason if raw.reason != "available" else "metric_not_exposed", unit=unit)
    return result


def parse_server(source: object) -> dict[str, object]:
    source = mapping(source)
    states = source.get("internal_states")
    state = mapping(states[0]) if isinstance(states, list) and len(states) == 1 else {}
    reason = "multiple_states_ambiguous" if isinstance(states, list) and len(states) > 1 else "counter_not_exposed"
    return {"settings": parse_settings(source.get("server_args", source)),
            "server_version": version(source.get("version")),
            "lifetime_accept_length": number(state.get("avg_spec_accept_length"), "tokens/request_verification_lifetime", reason),
            "server_counters": {"accepted_emitted_tokens": number(state.get("spec_total_num_accept_tokens"), "tokens", reason),
                                "request_verifications": number(state.get("spec_total_num_forward_ct"), "request_verifications", reason)}}


def collect_endpoint(endpoint: str, token: str | None = None,
                     identities: dict[str, str] | None = None) -> dict[str, object]:
    server = fetch_endpoint(endpoint, "/get_server_info", token)
    decoded = safe_json(server)
    result = parse_server(decoded)
    result["server_status"] = field(True) if isinstance(decoded, dict) else field(reason=server.reason if server.reason != "available" else "invalid_format")
    result["metrics"] = parse_metrics(fetch_endpoint(endpoint, "/metrics", token), identities)
    return result


def at(source: object, *keys: str) -> Field:
    for key in keys:
        source = mapping(source).get(key)
    return source if isinstance(source, dict) and "value" in source else field()


def counter_delta(before: Field, after: Field, seconds: float, rate: bool = True, unit: str = "per_second") -> Field:
    if not math.isfinite(seconds) or seconds <= 0:
        return field(reason="invalid_interval", unit=unit)
    start, end = before["value"], after["value"]
    if type(start) not in (int, float) or type(end) not in (int, float):
        return field(reason="counter_unavailable", unit=unit)
    if end < start:
        return field(reason="counter_reset", unit=unit)
    return field((end - start) / seconds if rate else end - start, unit=unit)


def derive_interval(before: dict, after: dict, seconds: float,
                    identities_before: dict[str, str] | None = None,
                    identities_after: dict[str, str] | None = None) -> dict[str, object]:
    def elapsed(group: str) -> float:
        first = mapping(before.get("sample_times")).get(group)
        last = mapping(after.get("sample_times")).get(group)
        return last - first if type(first) in (int, float) and type(last) in (int, float) else seconds

    def delta(path: tuple[str, ...], rate: bool = True, unit: str = "per_second") -> Field:
        return counter_delta(at(before, *path), at(after, *path), elapsed(path[0]), rate, unit)

    vm = {key: delta(("host", "vm_counters", key), unit=unit) for key, unit in (
        ("swap_in_pages", "pages/s"), ("swap_out_pages", "pages/s"), ("page_in_kib", "KiB/s"),
        ("page_out_kib", "KiB/s"), ("oom_kills", "events/s"))}
    cpu = {key: delta(("host", "cpu_ticks", key), False) for key in ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")}
    busy = field(reason="counter_unavailable", unit="percent")
    if all(entry["value"] is not None for entry in cpu.values()):
        total = sum(entry["value"] for entry in cpu.values())
        if total > 0:
            busy = field(100 * (total - cpu["idle"]["value"] - cpu["iowait"]["value"]) / total, unit="percent")
    cgroups: dict[str, object] = {}
    for group in ("cgroup", "container_cgroup"):
        cgroups[group] = {
            "cpu": {key: delta((group, "cpu_counters", key), False, "us" if key.endswith("_us") else "count")
                    for key in ("periods", "throttled_periods", "throttled_us", "usage_us")},
            "memory": {key: delta((group, "memory_events", key), False, "count") for key in ("low", "high", "max", "oom", "oom_kill", "limit_failures")}}
    gpu_events = {gpu: {event: delta(("gpu", "devices", gpu, "event_counters_us", event), False, "us") for event in EVENTS}
                  for gpu in mapping(mapping(after.get("gpu")).get("devices"))}
    pressure = {resource: {kind: delta(("host", "pressure", resource, f"{kind}_total_us"), False, "us")
                          for kind in ("some", "full")} for resource in ("cpu", "memory", "io")}
    accepted = delta(("endpoint", "server_counters", "accepted_emitted_tokens"), False, "tokens")
    verifies = delta(("endpoint", "server_counters", "request_verifications"), False, "request_verifications")
    yield_value = field(reason="yield_unavailable")
    yield_source = "unknown"
    if accepted["value"] is not None and verifies["value"] is not None and verifies["value"] > 0:
        yield_value = field(accepted["value"] / verifies["value"], unit="tokens/request_verification")
        yield_source = "interval_server_counters"
    elif accepted["reason"] != "counter_reset" and verifies["reason"] != "counter_reset":
        yield_value = at(after, "endpoint", "metrics", "accept_length")
        if yield_value["value"] is not None:
            yield_source = "reported_gauge_not_interval_measurement"
    metrics_rates = {}
    for key in ("generation_tokens", "request_verifications", "cached_tokens"):
        entry = delta(("endpoint", "metrics", key))
        if identities_before is not None and identities_after is not None and (
                key not in identities_before or key not in identities_after or identities_before[key] != identities_after[key]):
            entry = field(reason="series_identity_changed")
        metrics_rates[key + "_per_second"] = entry
    request_rate = delta(("endpoint", "server_counters", "request_verifications"), unit="request_verifications/s")
    if request_rate["value"] is not None:
        metrics_rates["request_verifications_per_second"] = request_rate
    estimated = field(reason="steady_full_batch_assumptions_unverified", unit="estimated_batches/s")
    running = at(after, "endpoint", "metrics", "running_requests")["value"]
    throughput = at(after, "endpoint", "metrics", "generation_throughput")["value"]
    if (type(running) in (int, float) and running > 0 and running == at(before, "endpoint", "metrics", "running_requests")["value"]
            and type(throughput) in (int, float) and type(yield_value["value"]) in (int, float) and yield_value["value"] > 0):
        estimated = field(throughput / (running * yield_value["value"]), unit="estimated_batches/s")
        estimated["reason"] = "estimator_requires_steady_decode_full_batches"
    global_cadence = field(reason="global_batch_counter_not_exposed", unit="iterations/s")
    global_cadence["status"] = "not_measured"
    return {"seconds": field(seconds, unit="s"), "host_vm_rates": vm, "host_cpu_busy_percent": busy,
            "cgroup_deltas": cgroups, "gpu_event_deltas_us": gpu_events, "pressure_deltas_us": pressure,
            "endpoint": {**metrics_rates, "accepted_emitted_yield": yield_value, "yield_source": yield_source,
                         "global_verifier_iterations_per_second": global_cadence, "estimated_batch_cadence_hz": estimated}}


COMPARISON_PATHS = {
    "tp_size": ("endpoint", "settings", "tp_size"),
    "speculative_num_steps": ("endpoint", "settings", "speculative_num_steps"),
    "speculative_eagle_topk": ("endpoint", "settings", "speculative_eagle_topk"),
    "speculative_num_draft_tokens": ("endpoint", "settings", "speculative_num_draft_tokens"),
    "context_length": ("endpoint", "settings", "context_length"),
    "running_requests": ("endpoint", "metrics", "running_requests"),
    "generation_throughput_reported_tps": ("endpoint", "metrics", "generation_throughput"),
    "accept_length_reported": ("endpoint", "metrics", "accept_length"),
    "cache_hit_rate": ("endpoint", "metrics", "cache_hit_rate"),
    "cpu_limit_cores": ("cgroup", "cpu_limit_cores"),
    "memory_limit_bytes": ("cgroup", "memory_limit_bytes"),
}


def finite_numeric(value: object) -> float | int | None:
    return value if type(value) in (int, float) and 0 <= value <= 1e20 and math.isfinite(value) else None


def comparison_values(snapshot: dict) -> dict[str, float | int | None]:
    return {key: finite_numeric(at(snapshot, *path)["value"]) for key, path in COMPARISON_PATHS.items()}


def compare_baseline(path: Path, current: dict[str, object]) -> dict[str, object]:
    raw = read_bounded(path)
    source = mapping(safe_json(raw))
    values = mapping(source.get("comparison_values"))
    result: dict[str, object] = {}
    for key in COMPARISON_PATHS:
        before, after = finite_numeric(values.get(key)), finite_numeric(current.get(key))
        difference = after - before if before is not None and after is not None else None
        percent = 100 * difference / before if difference is not None and before > 0 else None
        percent_reason = "missing_or_zero_baseline"
        if percent is not None and not math.isfinite(percent):
            percent = None
            percent_reason = "nonfinite_percentage"
        result[key] = {"baseline": field(before), "current": field(after), "difference": field(difference),
                       "change_percent": field(percent, reason=percent_reason, unit="percent")}
    return {"status": field(True) if values else field(reason=raw.reason if raw.reason != "available" else "invalid_baseline"), "fields": result}


class SafeParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.exit(2, "Invalid diagnostic options; use --help. Supplied values are not echoed.\n")


def gpu_selection(value: str) -> list[int]:
    if not re.fullmatch(r"[0-9]{1,4}(?:,[0-9]{1,4}){0,15}", value):
        raise ValueError("invalid selection")
    selected = [int(part) for part in value.split(",")]
    if len(selected) != len(set(selected)) or any(index > 4095 for index in selected):
        raise ValueError("invalid selection")
    return selected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = SafeParser(prog="collect_diagnostics", description="Read-only local diagnostics; endpoints and containers require explicit opt-in.", allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; readable report is FILE.txt (neither overwritten)")
    parser.add_argument("--gpus", type=gpu_selection, help="Up to 16 numeric local indices, in synthetic GPU order")
    parser.add_argument("--duration", type=float, default=0, help="Existing-load sampling window in seconds, 0..60; no workload generation")
    parser.add_argument("--interval", type=float, default=1, help="Sampling interval 0.25..10 seconds; at most 120 intervals")
    parser.add_argument("--container", help="Explicit existing local Docker container (not emitted)")
    parser.add_argument("--endpoint", type=validate_endpoint, help="Explicit HTTP(S) origin; GET server info and metrics only")
    parser.add_argument("--token-env", help="Name of environment variable holding bearer token; never pass token as an argument")
    parser.add_argument("--baseline", type=Path, help="Bounded prior JSON with numeric comparison_values only")
    args = parser.parse_args(argv)
    if (not math.isfinite(args.duration) or not 0 <= args.duration <= 60
            or not math.isfinite(args.interval) or not .25 <= args.interval <= 10
            or args.duration / args.interval > 120):
        parser.error("invalid bounds")
    if args.container is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", args.container):
        parser.error("invalid container")
    if args.token_env is not None and (args.endpoint is None or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", args.token_env)):
        parser.error("invalid token option")
    return args


CHECKLIST = [
    "Match pure decode versus end-to-end (E2E) throughput and units.",
    "Match single-replica versus fleet concurrency, TP size and exact GPU placement.",
    "Match exact marketed card edition, VRAM, configured power, NUMA and pair topology.",
    "Match input/output context, chunking, KV/Mamba capacity and cache hit/miss protocol.",
    "Match speculation steps, topk, draft count, head gate and observed acceptance.",
    "Match actual reasoning/emitted tokens, EOS behavior, output caps and counting policy.",
    "Match warmup/JIT/graph capture, workload timing and existing competing activity.",
]
CAVEATS = [
    "GPU utilization/activity is not proof of saturation or a bottleneck.",
    "P2P capability true is not measured bandwidth; no P2P benchmark is performed.",
    "Topology NODE crosses host bridges within a NUMA node; PHB traverses a host bridge. Neither measures latency or bandwidth.",
    "PCIe current generation/width may downshift while idle; compare current AND maximum under the same existing load.",
    "Power-limit clock events are not necessarily thermal events; compare limits, clocks, temperature and interval counters.",
    "Driver CUDA supported version is not the loaded runtime CUDA version. Package metadata is not proof of loaded libraries.",
    "Snapshots are sequential, not atomic; interval rates use observed collection times, not since-boot averages.",
    "Counter resets and multiple metric series are unavailable, not zeros. No blind sums across TP ranks or replicas.",
    "Accepted emitted-token yield per request verification differs from global verifier batch cadence. Cadence estimates require steady pure decode and full batches.",
    "Completion-accounted endpoint token/verification counters can lag actual decode and include mixed workloads; they are not a benchmark.",
    "This diagnostic establishes no cause, regression, or optimization fix for another workload. Do not automatically reboot or override P2P.",
    "Unsupported fields remain unavailable. Review both output files before sharing; hardware/configuration can still fingerprint a system.",
]


def discover_gpus() -> tuple[list[int], Field]:
    raw = run_bounded(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"])
    if raw.reason != "available":
        return [], field(reason=raw.reason)
    lines = raw.text.splitlines()
    if not lines or len(lines) > 16 or not all(re.fullmatch(r"[0-9]{1,4}", line.strip()) for line in lines):
        return [], field(reason="invalid_or_too_many_gpus_use_selection")
    try:
        selected = gpu_selection(",".join(line.strip() for line in lines))
    except ValueError:
        return [], field(reason="invalid_gpu_inventory")
    return selected, field(len(selected), unit="devices")


def capture_sample(selected: list[int], epoch: float, endpoint: str | None, token: str | None,
                   include_topology: bool = False, container_subject: str | None = None) -> tuple[dict, dict[str, str]]:
    sample: dict = {"sample_times": {}}
    for group, collector in (("host", collect_host), ("cgroup", collect_cgroup),
                             ("gpu", lambda: collect_gpus(selected, include_topology) if selected else {"devices": {}})):
        sample[group] = collector()
        sample["sample_times"][group] = time.monotonic() - epoch
    if container_subject:
        sample["container_cgroup"] = collect_cgroup(subject=container_subject)
        sample["sample_times"]["container_cgroup"] = time.monotonic() - epoch
    identities: dict[str, str] = {}
    if endpoint:
        sample["endpoint"] = collect_endpoint(endpoint, token, identities)
        sample["sample_times"]["endpoint"] = time.monotonic() - epoch
    sample["elapsed_seconds"] = time.monotonic() - epoch
    return sample, identities


def make_findings(snapshot: dict) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for gpu, device in mapping(mapping(snapshot.get("gpu")).get("devices")).items():
        findings.append({"code": gpu + "_observations", "evidence": device,
                         "caveat": "Observed values only; activity does not establish saturation or a root cause."})
    findings.append({"code": "host_constraints", "evidence": snapshot.get("cgroup", {}),
                     "caveat": "Limits apply to the collector cgroup, not automatically the serving process. Opt in to a container for its limits."})
    if "endpoint" in snapshot:
        findings.append({"code": "serving_observations", "evidence": snapshot["endpoint"],
                         "caveat": "Endpoint-to-GPU placement is not authenticated by this collector. Multiple series remain ambiguous."})
    return findings


def build_report(args: argparse.Namespace, token: str | None) -> dict[str, object]:
    selected, discovery = discover_gpus() if args.gpus is None else (args.gpus, field(len(args.gpus), unit="requested_devices"))
    epoch = time.monotonic()
    container_context: dict[str, str] = {}
    container = collect_container(args.container, container_context) if args.container else field(reason="not_opted_in")
    subject = container_context.get("subject")
    snapshot, identities = capture_sample(selected, epoch, args.endpoint, token, True, subject)
    samples = [snapshot]
    intervals = []
    start = time.monotonic()
    deadline = start + args.duration
    for index in range(1, math.ceil(args.duration / args.interval) + 1):
        now = time.monotonic()
        if now >= deadline:
            break
        target = min(start + index * args.interval, deadline)
        time.sleep(max(0, target - now))
        next_sample, next_identities = capture_sample(selected, epoch, args.endpoint, token, container_subject=subject)
        intervals.append(derive_interval(snapshot, next_sample,
                                         next_sample["elapsed_seconds"] - snapshot["elapsed_seconds"],
                                         identities, next_identities))
        samples.append(next_sample)
        snapshot, identities = next_sample, next_identities
    values = comparison_values(snapshot)
    return {"schema_version": 1, "gpu_discovery": discovery, "container": container,
            "endpoint": field(bool(args.endpoint)) if args.endpoint else field(reason="not_opted_in"),
            "samples": samples, "intervals": intervals, "comparison_values": values,
            "sampling": {"requested_duration_seconds": field(args.duration, unit="s"),
                         "interval_seconds": field(args.interval, unit="s"),
                         "actual_window_seconds": field(samples[-1]["elapsed_seconds"] - samples[0]["elapsed_seconds"], unit="s")},
            "baseline": compare_baseline(args.baseline, values) if args.baseline else field(reason="not_requested"),
            "findings": make_findings(snapshot), "caveats": CAVEATS, "benchmark_contract": CHECKLIST}


def render_report(report: dict) -> str:
    lines = ["Read-only diagnostics (not a benchmark or root-cause verdict)",
             f"Samples: {len(report['samples'])}; intervals: {len(report['intervals'])}", "", "Findings and evidence"]
    for finding in report["findings"]:
        lines.extend([finding["code"], json.dumps(finding["evidence"], indent=2, allow_nan=False), finding["caveat"], ""])
    lines.extend(["Interval evidence", json.dumps(report["intervals"], indent=2, allow_nan=False),
                  "Numerical baseline (not a matched benchmark verdict)", json.dumps(report["baseline"], indent=2, allow_nan=False),
                  "Caveats", *["- " + text for text in report["caveats"]], "Benchmark contract", *["- " + text for text in report["benchmark_contract"]]])
    return "\n".join(lines) + "\n"


def write_new(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(text)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_text = Path(str(args.output) + ".txt")
    if args.output.exists() or output_text.exists():
        print("Output already exists; choose new output files. Paths are not echoed.", file=sys.stderr)
        return 2
    token = os.environ.get(args.token_env) if args.token_env else None
    if args.token_env and not token:
        print("Requested token environment variable is empty or unavailable.", file=sys.stderr)
        return 2
    try:
        report = build_report(args, token)
        write_new(args.output, json.dumps(report, indent=2, allow_nan=False) + "\n")
        write_new(output_text, render_report(report))
    except (OSError, ValueError, RecursionError):
        print("Diagnostic collection or output failed; details suppressed for privacy. Partial output may exist.", file=sys.stderr)
        return 1
    print("Diagnostic JSON and adjacent readable report saved. Review both before sharing.")
    return 0


@dataclass(frozen=True)
class Raw:
    text: str = ""
    reason: str = "available"


def read_bounded(path: str | Path, limit: int = MAX_BYTES) -> Raw:
    try:
        with open(path, "rb") as stream:
            data = stream.read(limit + 1)
        return Raw(reason="oversized") if len(data) > limit else Raw(data.decode("utf-8", errors="replace"))
    except PermissionError:
        return Raw(reason="permission_denied")
    except FileNotFoundError:
        return Raw(reason="not_found")
    except OSError:
        return Raw(reason="unavailable")


def run_bounded(argv: list[str], timeout: float = TIMEOUT, limit: int = MAX_BYTES,
                env: dict[str, str] | None = None) -> Raw:
    try:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL, shell=False, env=env)
    except FileNotFoundError:
        return Raw(reason="tool_missing")
    except PermissionError:
        return Raw(reason="permission_denied")
    except OSError:
        return Raw(reason="unavailable")
    chunks: list[bytes] = []

    def drain() -> None:
        try:
            chunks.append(process.stdout.read(limit + 1))
            if len(chunks[0]) > limit:
                process.kill()
        except OSError:
            pass

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    reason = "available"
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        reason = "timeout"
        process.kill()
        process.wait()
    reader.join(timeout=timeout)
    process.stdout.close()
    data = chunks[0] if chunks else b""
    if len(data) > limit:
        return Raw(reason="oversized")
    if reason != "available":
        return Raw(reason=reason)
    if process.returncode:
        return Raw(reason="command_failed")
    return Raw(data.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    raise SystemExit(http_worker_main() if sys.argv[1:] == ["--http-worker"] else main())

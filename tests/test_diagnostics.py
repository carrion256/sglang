"""Read-only diagnostic collector behavior and privacy regressions."""
import importlib.util
import http.client
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/collect_diagnostics.py"
if SCRIPT.exists():
    SPEC = importlib.util.spec_from_file_location("diagnostics", SCRIPT)
    diagnostics = importlib.util.module_from_spec(SPEC)
    sys.modules[SPEC.name] = diagnostics
    SPEC.loader.exec_module(diagnostics)
else:
    diagnostics = types.SimpleNamespace()


class DiagnosticsTests(unittest.TestCase):
    def api(self, name):
        self.assertTrue(callable(getattr(diagnostics, name, None)), name + " missing")
        return getattr(diagnostics, name)

    def test_http_protocol_errors_never_escape_or_reach_stderr(self):
        errors = (http.client.BadStatusLine("PRIVATE_SYNTHETIC_RESPONSE"),
                  http.client.IncompleteRead(b"PRIVATE_SYNTHETIC_RESPONSE", 99))
        for error in errors:
            with self.subTest(error=type(error).__name__):
                stderr = io.StringIO()
                with patch.object(diagnostics.urllib.request, "build_opener") as build, patch("sys.stderr", stderr):
                    build.return_value.open.side_effect = error
                    result = diagnostics.fetch_endpoint_worker("http://localhost", "/metrics")
                self.assertEqual(result.reason, "connection_failed")
                self.assertEqual(result.text, "")
                self.assertEqual(stderr.getvalue(), "")

    def test_baseline_hundreds_digit_integer_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.json"
            baseline.write_text(json.dumps({"comparison_values": {"cpu_limit_cores": 10**400}}))
            result = diagnostics.compare_baseline(baseline, {"cpu_limit_cores": 2})
        values = result["fields"]["cpu_limit_cores"]
        self.assertEqual(values["baseline"]["status"], "unavailable")
        self.assertEqual(values["difference"]["status"], "unavailable")
        self.assertEqual(values["current"]["value"], 2)
        self.assertIsNone(diagnostics.finite_numeric(-(10**400)))

    def test_baseline_tiny_positive_value_percentage_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.json"
            baseline.write_text(json.dumps({"comparison_values": {"cache_hit_rate": 1e-320}}))
            result = diagnostics.compare_baseline(baseline, {"cache_hit_rate": 0.5})
        json.dumps(result, allow_nan=False)
        values = result["fields"]["cache_hit_rate"]
        for name, expected in (("baseline", 1e-320), ("current", 0.5), ("difference", 0.5 - 1e-320)):
            self.assertEqual(values[name]["status"], "available")
            self.assertEqual(values[name]["value"], expected)
        self.assertEqual(values["change_percent"]["status"], "unavailable")
        self.assertIsNone(values["change_percent"]["value"])
        self.assertEqual(values["change_percent"]["reason"], "nonfinite_percentage")

    def test_docker_remote_environment_and_default_context_cannot_override_local_socket(self):
        remote = {"DOCKER_HOST": "tcp://PRIVATE_REMOTE:2376", "DOCKER_CONTEXT": "PRIVATE_REMOTE",
                  "DOCKER_TLS_VERIFY": "1", "DOCKER_CERT_PATH": "/PRIVATE_CERTS"}
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text('{"currentContext":"PRIVATE_REMOTE"}')
            remote["DOCKER_CONFIG"] = directory
            with patch.dict(os.environ, remote), patch.object(diagnostics, "run_bounded") as run:
                run.side_effect = [diagnostics.Raw('[{"State":{"Pid":0}}]'), diagnostics.Raw('{}')]
                diagnostics.collect_container("serving")
                self.assertEqual(run.call_count, 2)
                for call in run.call_args_list:
                    argv = call.args[0]
                    self.assertEqual(argv[:3], ["docker", "--host", "unix:///var/run/docker.sock"])
                    self.assertEqual(argv[3], "--config")
                    self.assertNotEqual(argv[4], directory)
                    self.assertFalse(any(key.startswith("DOCKER_") for key in call.kwargs["env"]))

    def test_docker_cgroup_does_not_guess_host_pid_inside_collector_container(self):
        context = {}
        with patch.object(diagnostics, "run_bounded") as run, \
                patch.object(diagnostics.Path, "exists", return_value=True), \
                patch.object(diagnostics, "collect_cgroup") as collect:
            run.side_effect = [diagnostics.Raw('[{"State":{"Pid":1234}}]'), diagnostics.Raw('{}')]
            result = diagnostics.collect_container("serving", context)
        collect.assert_not_called()
        self.assertEqual(context, {})
        self.assertEqual(result["cgroup"]["status"]["reason"], "container_pid_namespace_unverified")

    def test_docker_pid_namespace_requires_visible_peer_and_matching_namespaces(self):
        for peer, namespaces, expected in [(0, [], False), (42, ["pid-host", "pid-other"], False),
                                           (42, ["pid-host", "pid-host", "mnt-host", "mnt-other"], False),
                                           (42, ["pid-host", "pid-host", "mnt-host", "mnt-host"], True)]:
            with self.subTest(peer=peer, namespaces=namespaces), \
                    patch.object(diagnostics.Path, "exists", return_value=False), \
                    patch.object(diagnostics.socket, "socket") as connection, \
                    patch.object(diagnostics.os, "readlink", side_effect=namespaces):
                connection.return_value.__enter__.return_value.getsockopt.return_value = struct.pack("3i", peer, 0, 0)
                self.assertEqual(diagnostics.docker_pid_namespace_verified(), expected)
                connection.return_value.__enter__.return_value.connect.assert_called_once_with("/var/run/docker.sock")
        with patch.object(diagnostics.Path, "exists", return_value=False), \
                patch.object(diagnostics.socket, "socket", side_effect=PermissionError("PRIVATE_SENTINEL")):
            self.assertFalse(diagnostics.docker_pid_namespace_verified())

    @contextmanager
    def slow_http_origin(self, mode, received=None):
        stop = threading.Event()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(3)

            def serve():
                try:
                    with listener.accept()[0] as connection:
                        connection.settimeout(3)
                        request = connection.recv(16384)
                        if received is not None:
                            received.append(request)
                        if mode == "malformed":
                            connection.sendall(b"PRIVATE_SYNTHETIC_RESPONSE\r\n\r\n")
                            return
                        if mode == "success":
                            body = b'sglang:num_queue_reqs 7\nPRIVATE_SENTINEL{secret="SECRET"} 42\n'
                            connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
                            return
                        if mode == "headers":
                            connection.sendall(b"HTTP/1.1 200 OK\r\nX-Stall: ")
                            for _ in range(50):
                                if stop.wait(.04):
                                    return
                                connection.sendall(b"x")
                            connection.sendall(b"\r\nContent-Length: 0\r\n\r\n")
                        else:
                            if stop.wait(.5):
                                return
                            connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n")
                            for chunk in (b"a", b"b"):
                                if stop.wait(.5):
                                    return
                                connection.sendall(chunk)
                except OSError:
                    pass

            thread = threading.Thread(target=serve, daemon=True)
            thread.start()
            try:
                yield "http://127.0.0.1:" + str(listener.getsockname()[1])
            finally:
                stop.set()
                thread.join(timeout=4)
                self.assertFalse(thread.is_alive())

    def test_http_total_deadline_terminates_trickled_headers(self):
        with self.slow_http_origin("headers") as endpoint:
            start = time.monotonic()
            result = diagnostics.fetch_endpoint(endpoint, "/metrics", timeout=.6)
            elapsed = time.monotonic() - start
        self.assertEqual(result.reason, "timeout")
        self.assertLess(elapsed, .9)

    def test_http_total_deadline_includes_headers_and_slow_body(self):
        with self.slow_http_origin("body") as endpoint:
            start = time.monotonic()
            result = diagnostics.fetch_endpoint(endpoint, "/metrics", timeout=.6)
            elapsed = time.monotonic() - start
        self.assertEqual(result.reason, "timeout")
        self.assertLess(elapsed, .9)

    def test_http_total_deadline_kills_and_reaps_stalled_dns_worker(self):
        processes = []
        popen = subprocess.Popen
        bootstrap = ("import runpy,socket,sys,time; "
                     "socket.getaddrinfo=lambda *args,**kwargs: time.sleep(10); "
                     "sys.argv=[sys.argv[1],'--http-worker']; "
                     "runpy.run_path(sys.argv[0],run_name='__main__')")

        def launch(argv, **kwargs):
            self.assertEqual(argv[1:4], ["-I", "-S", "-B"])
            process = popen([sys.executable, "-I", "-S", "-B", "-c", bootstrap, str(SCRIPT)], **kwargs)
            processes.append(process)
            return process

        with patch.object(diagnostics.subprocess, "Popen", side_effect=launch):
            start = time.monotonic()
            result = diagnostics.fetch_endpoint("http://never-resolved.invalid", "/metrics", timeout=.6)
            elapsed = time.monotonic() - start
        self.assertEqual(result.reason, "timeout")
        self.assertLess(elapsed, .9)
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].poll())
        self.assertTrue(processes[0].stdout.closed)
        self.assertTrue(processes[0].stdin.closed)

    def test_http_worker_inherits_auth_without_command_line_secrets_and_preserves_metrics(self):
        received = []
        with self.slow_http_origin("success", received) as endpoint, \
                patch.object(diagnostics.subprocess, "Popen", wraps=subprocess.Popen) as launch, \
                patch("sys.stderr", new_callable=io.StringIO) as stderr:
            result = diagnostics.fetch_endpoint(endpoint, "/metrics", "PRIVATE_AUTH")
        self.assertEqual(result.reason, "available")
        self.assertIn(b"Authorization: Bearer PRIVATE_AUTH", received[0])
        self.assertEqual(launch.call_args.kwargs["env"]["DIAGNOSTIC_HTTP_TOKEN"], "PRIVATE_AUTH")
        self.assertNotIn("PRIVATE_AUTH", str(launch.call_args.args))
        self.assertNotIn(endpoint, str(launch.call_args.args))
        self.assertEqual(stderr.getvalue(), "")
        metrics = diagnostics.parse_metrics(result)
        self.assertEqual(metrics["queued_requests"]["value"], 7)
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(metrics))

    def test_http_real_malformed_status_has_no_traceback_or_endpoint_output(self):
        with self.slow_http_origin("malformed") as endpoint, \
                patch("sys.stderr", new_callable=io.StringIO) as stderr, \
                patch("sys.stdout", new_callable=io.StringIO) as stdout:
            result = diagnostics.fetch_endpoint(endpoint, "/metrics")
        self.assertEqual(result.reason, "connection_failed")
        self.assertEqual(result.text, "")
        self.assertEqual(stderr.getvalue() + stdout.getvalue(), "")

    def test_http_parent_allowlists_worker_errors_and_suppresses_launch_errors(self):
        for output in (b'{"text":"PRIVATE_SENTINEL","reason":"http_error"}',
                       b'{"text":"PRIVATE_SENTINEL","reason":"PRIVATE_ENDPOINT"}', b'PRIVATE_ENDPOINT'):
            with self.subTest(output=output), patch.object(diagnostics.subprocess, "Popen") as launch:
                process = launch.return_value.__enter__.return_value
                process.returncode = 0
                process.communicate.return_value = (output, None)
                result = diagnostics.fetch_endpoint("http://localhost", "/metrics")
                self.assertIsNone(launch.call_args.kwargs["env"].get("DIAGNOSTIC_HTTP_TOKEN"))
            self.assertEqual(result.text, "")
            self.assertIn(result.reason, ("http_error", "connection_failed"))
        with patch.object(diagnostics.subprocess, "Popen", side_effect=OSError("PRIVATE_ENDPOINT")):
            self.assertEqual(diagnostics.fetch_endpoint("http://localhost", "/metrics").reason, "connection_failed")
        with patch.object(diagnostics.subprocess, "Popen") as launch:
            self.assertEqual(diagnostics.fetch_endpoint("http://localhost", "/metrics", "SECRET\n").reason, "invalid_token")
            with self.assertRaises(ValueError):
                diagnostics.fetch_endpoint("http://localhost", "/v1/chat/completions")
            launch.assert_not_called()

    def test_http_worker_is_reaped_even_when_pipe_communication_fails(self):
        with patch.object(diagnostics.subprocess, "Popen") as launch:
            process = launch.return_value.__enter__.return_value
            process.communicate.side_effect = OSError("PRIVATE_ENDPOINT")
            process.poll.return_value = None
            result = diagnostics.fetch_endpoint("http://localhost", "/metrics")
        self.assertEqual(result.reason, "connection_failed")
        process.kill.assert_called_once_with()
        process.wait.assert_called_once_with()

    def test_bounded_local_reads_and_command_failures_are_sanitized(self):
        read = self.api("read_bounded")
        run = self.api("run_bounded")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "PRIVATE_SENTINEL"
            path.write_text("123456789")
            self.assertEqual(read(path, limit=4).reason, "oversized")
            self.assertEqual(read(path, limit=20).text, "123456789")
            self.assertEqual(read(path / "absent").reason, "unavailable")
        with patch("builtins.open", side_effect=PermissionError("PRIVATE_SENTINEL")):
            self.assertEqual(read("/proc/meminfo").reason, "permission_denied")
        self.assertEqual(run(["/definitely-missing-diagnostic-tool"]).reason, "tool_missing")
        with patch.object(diagnostics.subprocess, "Popen", side_effect=PermissionError("PRIVATE_SENTINEL")):
            self.assertEqual(run(["nvidia-smi"]).reason, "permission_denied")
        self.assertEqual(run([sys.executable, "-c", "import time; time.sleep(2)"], timeout=0.05).reason, "timeout")
        self.assertEqual(run([sys.executable, "-c", "print('x'*4096)"], limit=64).reason, "oversized")
        self.assertEqual(run([sys.executable, "-c", "raise SystemExit(1)"]).reason, "command_failed")

    def test_gpu_xml_versions_units_and_identifier_redaction(self):
        parse = self.api("parse_gpu")
        for power_tag, clock_tag in [("power_readings", "clocks_throttle_reasons"),
                                     ("gpu_power_readings", "clocks_event_reasons")]:
            xml = f'''<?xml version="1.0"?><!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v12.dtd">
            <nvidia_smi_log><driver_version>580.82.07</driver_version><cuda_version>13.0</cuda_version>
            <hostname>PRIVATE_SENTINEL</hostname><gpu id="0000:01:00.0"><product_name>NVIDIA RTX PRO 6000 Blackwell Workstation Edition</product_name>
            <uuid>GPU-PRIVATE_SENTINEL</uuid><serial>PRIVATE_SENTINEL</serial>
            <fb_memory_usage><total>97887 MiB</total></fb_memory_usage>
            <pci><pci_gpu_link_info><pcie_gen><current_link_gen>1</current_link_gen><max_link_gen>5</max_link_gen></pcie_gen>
            <link_widths><current_link_width>16x</current_link_width><max_link_width>16x</max_link_width></link_widths></pci_gpu_link_info></pci>
            <{power_tag}><power_draw>123.50 W</power_draw><power_limit>600.00 W</power_limit><default_power_limit>600 W</default_power_limit><max_power_limit>600 W</max_power_limit></{power_tag}>
            <clocks><sm_clock>2400 MHz</sm_clock><mem_clock>14000 MHz</mem_clock></clocks>
            <temperature><gpu_temp>45 C</gpu_temp></temperature><performance_state>P2</performance_state>
            <utilization><gpu_util>96 %</gpu_util><memory_util>N/A</memory_util></utilization>
            <{clock_tag}><{clock_tag}_sw_power_cap>Active</{clock_tag}_sw_power_cap></{clock_tag}>
            <clocks_event_reasons_counters><sw_power_cap>150 us</sw_power_cap></clocks_event_reasons_counters>
            </gpu></nvidia_smi_log>'''
            result = parse(diagnostics.Raw(xml))
            self.assertEqual(result["model"]["value"], "NVIDIA RTX PRO 6000 Blackwell Workstation Edition")
            self.assertEqual(result["power_w"]["value"], 123.5)
            self.assertEqual(result["pcie_max_generation"]["value"], 5)
            self.assertEqual(result["pcie_current_width"]["value"], 16)
            self.assertEqual(result["memory_util_percent"]["status"], "unavailable")
            self.assertTrue(result["clock_events"]["sw_power_cap"]["value"])
            self.assertEqual(result["event_counters_us"]["sw_power_cap"]["value"], 150)
            self.assertNotIn("PRIVATE_SENTINEL", json.dumps(result))
            self.assertNotIn("0000:", json.dumps(result))
        for raw in [diagnostics.Raw(reason="timeout"), diagnostics.Raw("<broken>"),
                    diagnostics.Raw('<!DOCTYPE x [<!ENTITY x "PRIVATE_SENTINEL">]><x>&x;</x>')]:
            self.assertIsNone(parse(raw)["model"]["value"])
        hostile = diagnostics.Raw('<nvidia_smi_log><driver_version>1.2/PRIVATE_SENTINEL</driver_version><gpu><product_name>NVIDIA PRIVATE_SENTINEL</product_name></gpu></nvidia_smi_log>')
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(parse(hostile)))

    def test_selected_gpu_collection_filters_topology_and_capabilities(self):
        collect = self.api("collect_gpus")
        calls = []

        def command(argv, **kwargs):
            calls.append(argv)
            if "-x" in argv:
                return diagnostics.Raw('<nvidia_smi_log><gpu><product_name>NVIDIA GeForce RTX 4090</product_name></gpu></nvidia_smi_log>')
            if "-m" in argv:
                return diagnostics.Raw('GPU0 GPU2 GPU5 CPU Affinity NUMA Affinity\nGPU0 X SYS SYS 0-7 0\nGPU2 SYS X PHB 8-15 1\nGPU5 SYS PHB X 8-15 1\nLegend: PRIVATE_SENTINEL')
            if "-p2p" in argv:
                return diagnostics.Raw('GPU0 GPU2 GPU5\nGPU0 X OK CNS\nGPU2 OK X OK\nGPU5 CNS OK X\n')
            return diagnostics.Raw('CUDA Version: 13.0')

        with patch.object(diagnostics, "run_bounded", side_effect=command):
            result = collect([5, 2])
        self.assertEqual(list(result["devices"]), ["GPU0", "GPU1"])
        self.assertEqual(result["topology"]["GPU0"]["GPU1"]["value"], "PHB")
        self.assertTrue(result["p2p_read_capability"]["GPU0"]["GPU1"]["value"])
        self.assertEqual(result["devices"]["GPU0"]["driver_cuda_supported_version"]["value"], "13.0")
        self.assertNotIn("GPU5", json.dumps(result))
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(result))
        self.assertTrue(any(argv[-2:] == ["-i", "5"] for argv in calls))
        with patch.object(diagnostics, "run_bounded", return_value=diagnostics.Raw(reason="tool_missing")):
            absent = collect([2])
        self.assertEqual(absent["devices"]["GPU0"]["power_w"]["reason"], "tool_missing")
        self.assertIsNone(absent["p2p_read_capability"]["GPU0"]["GPU0"]["value"])
        parse = diagnostics.parse_matrix
        self.assertIsNone(parse(diagnostics.Raw('GPU2 GPU5\nGPU2 X PRIVATE_SENTINEL'), [2, 5])["GPU0"]["GPU1"]["value"])

    def test_host_collection_reads_numeric_topology_pressure_and_counters(self):
        collect = self.api("collect_host")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contents = {
                "proc/cpuinfo": "processor : 0\nmodel name : AMD Ryzen 9 9950X 16-Core Processor\nSerial : PRIVATE_SENTINEL\n",
                "proc/self/status": "Name: PRIVATE_SENTINEL\nCpus_allowed_list:\t0-3,8\nMems_allowed_list:\t0-1\n",
                "proc/meminfo": "MemTotal: 1024 kB\nMemAvailable: 512 kB\nSwapTotal: 64 kB\nSwapFree: 32 kB\n",
                "proc/vmstat": "pswpin 10\npswpout 20\npgpgin 300\npgpgout 400\noom_kill 1\nPRIVATE_SENTINEL 9\n",
                "proc/stat": "cpu 100 2 10 800 4 1 2 0 0 0\n",
                "proc/pressure/memory": "some avg10=1.00 avg60=2.00 avg300=3.00 total=400\nfull avg10=0.1 avg60=0.2 avg300=0.3 total=40\n",
                "sys/devices/system/node/online": "0-1",
                "sys/devices/system/node/node0/cpulist": "0-3",
                "sys/devices/system/node/node0/distance": "10 32",
                "sys/devices/system/node/node1/cpulist": "8-11",
                "sys/devices/system/node/node1/distance": "32 10",
            }
            for name, text in contents.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            result = collect(root / "proc", root / "sys")
        self.assertEqual(result["cpu_model"]["value"], "AMD Ryzen 9 9950X 16-Core Processor")
        self.assertEqual(result["cpuset"]["value"], [0, 1, 2, 3, 8])
        self.assertEqual(result["numa"]["node1"]["cpus"]["value"], [8, 9, 10, 11])
        self.assertEqual(result["memory_kib"]["available"]["value"], 512)
        self.assertEqual(result["vm_counters"]["swap_in_pages"]["value"], 10)
        self.assertEqual(result["pressure"]["memory"]["some_total_us"]["value"], 400)
        self.assertIsNone(result["pressure"]["cpu"]["some_total_us"]["value"])
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(result))
        self.assertIsNone(diagnostics.parse_cpu_list("0-999999999")["value"])
        self.assertIsNone(diagnostics.cpu_model("AMD PRIVATE_SENTINEL")["value"])

    def test_cgroup_v2_and_v1_limits_and_counters_without_paths(self):
        collect = self.api("collect_cgroup")
        for v2 in (True, False):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                proc = root / "proc"
                mount = root / "cgroup"
                group = mount / "PRIVATE_SENTINEL"
                group.mkdir(parents=True)
                (proc / "self").mkdir(parents=True)
                (proc / "self/cgroup").write_text("0::/PRIVATE_SENTINEL\n" if v2 else "2:cpu,cpuacct,memory,cpuset:/PRIVATE_SENTINEL\n")
                (proc / "self/mountinfo").write_text(f"1 0 0:1 / {mount} rw - {'cgroup2' if v2 else 'cgroup'} cgroup rw,cpu,cpuacct,memory,cpuset\n")
                values = ({"cpu.max": "200000 100000", "cpu.stat": "nr_periods 10\nnr_throttled 2\nthrottled_usec 30\nusage_usec 500\n",
                           "memory.max": "2048", "memory.current": "1024", "memory.swap.max": "max", "memory.events": "high 2\noom 1\noom_kill 0\n",
                           "cpuset.cpus.effective": "0-1", "cpuset.mems.effective": "0"} if v2 else
                          {"cpu.cfs_quota_us": "200000", "cpu.cfs_period_us": "100000", "cpu.stat": "nr_periods 10\nnr_throttled 2\nthrottled_time 30000\n",
                           "memory.limit_in_bytes": "2048", "memory.usage_in_bytes": "1024", "memory.failcnt": "3", "cpuset.cpus": "0-1", "cpuset.mems": "0"})
                for name, text in values.items():
                    (group / name).write_text(text)
                for name, text in ({"cpu.max": "100000 100000", "memory.max": "4096", "memory.swap.max": "max"} if v2 else
                                   {"cpu.cfs_quota_us": "100000", "cpu.cfs_period_us": "100000", "memory.limit_in_bytes": "4096"}).items():
                    (mount / name).write_text(text)
                result = collect(proc)
            self.assertEqual(result["version"]["value"], 2 if v2 else 1)
            self.assertEqual(result["cpu_limit_cores"]["value"], 1)
            self.assertEqual(result["memory_limit_bytes"]["value"], 2048)
            self.assertEqual(result["cpu_counters"]["throttled_us"]["value"], 30)
            self.assertEqual(result["cpuset"]["value"], [0, 1])
            self.assertNotIn("PRIVATE_SENTINEL", json.dumps(result))
            self.assertNotIn(directory, json.dumps(result))
        with patch.object(diagnostics, "read_bounded", return_value=diagnostics.Raw(reason="permission_denied")):
            self.assertIsNone(collect()["memory_limit_bytes"]["value"])

    def test_container_allowlists_values_digest_effective_args_and_fixed_probe(self):
        collect = self.api("collect_container")
        secret = "PRIVATE_SENTINEL /home/alice/private 192.168.1.2 GPU-abc-uuid alice@example.com"
        inspect = [{"Id": secret, "Name": secret, "Image": "sha256:" + "a" * 64,
                    "Args": ["-m", "sglang.launch_server", "--tp-size=2", "--speculative-num-steps", "3", "--speculative-eagle-topk=1", "--speculative-num-draft-tokens=4",
                             "--model-path", secret, "--attention-backend=" + secret, "--quantization=modelopt_mixed", "--disable-cuda-graph"],
                    "Config": {"Image": "private/" + secret, "Labels": {secret: {"version": secret}},
                               "Env": ["API_TOKEN=" + secret, "NCCL_SOCKET_IFNAME=private0", "NCCL_P2P_DISABLE=0", "SGLANG_PRIVATE_DRAFT_NVFP4_A16=1", "SGLANG_PLE_PACKED_NVFP4=1"],
                               "Cmd": ["--tp-size=8"]}, "HostConfig": {"ShmSize": 67108864, "Binds": [secret]},
                    "State": {"Pid": 0}, "NetworkSettings": {secret: secret}}]
        calls = []

        def command(argv, **kwargs):
            calls.append(argv)
            return diagnostics.Raw(json.dumps(inspect if "inspect" in argv else {"sglang": "0.5.3", "torch": "2.8.0+cu128", "cuda_runtime_package": "12.8.90", secret: secret}))

        with patch.object(diagnostics, "run_bounded", side_effect=command):
            result = collect("private-container")
        self.assertEqual(result["image_digest"]["value"], "sha256:" + "a" * 64)
        self.assertEqual(result["settings"]["tp_size"]["value"], 2)
        self.assertEqual(result["settings"]["speculative_num_steps"]["value"], 3)
        self.assertTrue(result["environment"]["SGLANG_PRIVATE_DRAFT_NVFP4_A16"]["value"])
        self.assertEqual(result["runtime_packages"]["sglang"]["value"], "0.5.3")
        self.assertEqual(result["shm_limit_bytes"]["value"], 67108864)
        for sentinel in [secret, "PRIVATE_SENTINEL", "private0", "private-container", "alice", "192.168"]:
            self.assertNotIn(sentinel, json.dumps(result))
        self.assertEqual(calls[1][:3], ["docker", "--host", "unix:///var/run/docker.sock"])
        self.assertEqual(calls[1][5:11], ["exec", "private-container", "python3", "-I", "-S", "-B"])
        self.assertNotIn("import torch", calls[1][-1])
        self.assertNotIn("import sglang", calls[1][-1])
        parse = diagnostics.parse_settings
        self.assertEqual(parse({"tp_size": {"evil": secret}, "quantization": secret})["tp_size"]["status"], "unavailable")
        self.assertIsNone(diagnostics.settings_from_args(["--tp-size=2", "--tp-size=8"])["tp_size"]["value"])
        with patch.object(diagnostics, "run_bounded", return_value=diagnostics.Raw(reason="permission_denied")):
            missing = collect("safe")
        self.assertIsNone(missing["image_digest"]["value"])
        self.assertIsNone(diagnostics.parse_container({"Image": "private:latest"})["image_digest"]["value"])

    def test_http_opt_in_rejects_credentials_redirects_and_oversize(self):
        fetch = self.api("fetch_endpoint_worker")
        for url in ["ftp://private/", "http://alice:SECRET@localhost", "http://localhost/?token=SECRET", "http://localhost/private", "http://localhost/#SECRET", "http://local host"]:
            with self.assertRaises(ValueError):
                diagnostics.validate_endpoint(url)
        self.assertEqual(diagnostics.validate_endpoint("http://127.0.0.1:30000/"), "http://127.0.0.1:30000")

        class Response(io.BytesIO):
            status = 200

        response = Response(b"safe")
        with patch.object(diagnostics.urllib.request, "build_opener") as build:
            opener = build.return_value
            opener.open.return_value = response
            raw = fetch("http://127.0.0.1:30000", "/metrics", "SECRET")
            request = opener.open.call_args.args[0]
            self.assertEqual(request.get_header("Authorization"), "Bearer SECRET")
            self.assertEqual(request.get_method(), "GET")
            self.assertEqual(raw.text, "safe")
            self.assertTrue(any(isinstance(handler, diagnostics.NoRedirect) for handler in build.call_args.args))
            self.assertTrue(any(isinstance(handler, diagnostics.urllib.request.ProxyHandler) and handler.proxies == {} for handler in build.call_args.args))
            response = Response(b"SECRET")
            response.status = 302
            opener.open.return_value = response
            self.assertEqual(fetch("http://localhost", "/metrics", "SECRET").reason, "redirect_refused")
            opener.open.return_value = Response(b"x" * 65)
            self.assertEqual(fetch("http://localhost", "/metrics", limit=64).reason, "oversized")
            opener.open.side_effect = TimeoutError("http://SECRET")
            self.assertEqual(fetch("http://localhost", "/metrics").reason, "timeout")
        self.assertIsNone(diagnostics.NoRedirect().redirect_request(None, None, 302, "SECRET", {}, "http://SECRET"))
        with self.assertRaises(ValueError):
            fetch("http://localhost", "/v1/chat/completions")

    def test_endpoint_typed_metrics_drop_nested_secrets_and_duplicate_series(self):
        collect = self.api("collect_endpoint")
        secret = 'PRIVATE_SENTINEL alice@192.168.1.2 /home/alice/checkpoint GPU-serial-uuid'
        server = {"server_args": {"tp_size": 2, "speculative_algorithm": "NEXTN", "speculative_num_steps": 3,
                                  "quantization": secret, "model_path": secret, secret: {"tp_size": secret}},
                  "internal_states": [{"avg_spec_accept_length": 2.5, "spec_total_num_accept_tokens": 100, "spec_total_num_forward_ct": 40, "private": secret}],
                  "version": "0.5.3", secret: secret}
        metrics = ('sglang:num_running_reqs{rank="0",model="' + secret + '"} 16\n'
                   'sglang:num_running_reqs{rank="1"} 16\n'
                   'sglang:num_queue_reqs 0\nsglang:spec_accept_length 2.5\n'
                   'sglang:cache_hit_rate 0.75\nsglang:mamba_usage 0.5\n'
                   'sglang:generation_tokens_total{model="' + secret + '"} 100\n'
                   'sglang:spec_verify_calls_total 40\nsglang:token_usage NaN\n'
                   'sglang:mamba_used_tokens{evil="escaped\\\"value"} 20\n'
                   'PRIVATE_SENTINEL 999\n')
        identities = {}
        with patch.object(diagnostics, "fetch_endpoint", side_effect=[diagnostics.Raw(json.dumps(server)), diagnostics.Raw(metrics)]):
            result = collect("http://localhost", identities=identities)
        self.assertEqual(result["settings"]["tp_size"]["value"], 2)
        self.assertEqual(result["metrics"]["running_requests"]["reason"], "multiple_series_ambiguous")
        self.assertEqual(result["metrics"]["queued_requests"]["value"], 0)
        self.assertEqual(result["metrics"]["mamba_used_tokens"]["value"], 20)
        self.assertEqual(result["server_counters"]["accepted_emitted_tokens"]["value"], 100)
        self.assertEqual(result["server_counters"]["request_verifications"]["value"], 40)
        self.assertIn("generation_tokens", identities)
        self.assertIsNone(result["metrics"]["token_usage"]["value"])
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(result))
        self.assertNotIn("alice", json.dumps(result))
        server["internal_states"].append(server["internal_states"][0])
        parsed = diagnostics.parse_server(server)
        self.assertIsNone(parsed["server_counters"]["request_verifications"]["value"])
        for hostile in [{"server_args": [secret]}, {"internal_states": [secret]}, [secret], None]:
            self.assertNotIn(secret, json.dumps(diagnostics.parse_server(hostile)))

    def test_interval_rates_counter_resets_and_yield_are_not_global_iterations(self):
        derive = self.api("derive_interval")
        value = diagnostics.field
        before = {"host": {"vm_counters": {"swap_in_pages": value(10)}},
                  "endpoint": {"server_counters": {"accepted_emitted_tokens": value(100), "request_verifications": value(40)},
                               "metrics": {"generation_tokens": value(100), "request_verifications": value(40),
                                           "running_requests": value(16), "generation_throughput": value(160), "accept_length": value(2.5)}}}
        after = {"host": {"vm_counters": {"swap_in_pages": value(20)}},
                 "endpoint": {"server_counters": {"accepted_emitted_tokens": value(180), "request_verifications": value(60)},
                              "metrics": {"generation_tokens": value(180), "request_verifications": value(60),
                                          "running_requests": value(16), "generation_throughput": value(160), "accept_length": value(4)}}}
        result = derive(before, after, 2.0)
        self.assertEqual(result["host_vm_rates"]["swap_in_pages"]["value"], 5)
        self.assertEqual(result["endpoint"]["accepted_emitted_yield"]["value"], 4)
        self.assertEqual(result["endpoint"]["yield_source"], "interval_server_counters")
        self.assertEqual(result["endpoint"]["request_verifications_per_second"]["value"], 10)
        self.assertEqual(result["endpoint"]["global_verifier_iterations_per_second"]["status"], "not_measured")
        self.assertEqual(result["endpoint"]["estimated_batch_cadence_hz"]["value"], 2.5)
        self.assertEqual(diagnostics.counter_delta(value(10), value(5), 1)["reason"], "counter_reset")
        self.assertEqual(diagnostics.counter_delta(value(10), value(20), 0)["reason"], "invalid_interval")
        self.assertIsNone(diagnostics.counter_delta(value(), value(5), 1)["value"])
        after["endpoint"]["server_counters"] = {}
        fallback = derive(before, after, 2.0)
        self.assertEqual(fallback["endpoint"]["yield_source"], "reported_gauge_not_interval_measurement")
        identities_before = {"generation_tokens": "PRIVATE_A"}
        identities_after = {"generation_tokens": "PRIVATE_B"}
        changed = derive(before, after, 2.0, identities_before, identities_after)
        self.assertEqual(changed["endpoint"]["generation_tokens_per_second"]["reason"], "series_identity_changed")

    def test_baseline_compares_only_named_numeric_values(self):
        compare = self.api("compare_baseline")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "PRIVATE_SENTINEL.json"
            path.write_text(json.dumps({"comparison_values": {"tp_size": 2, "generation_throughput_reported_tps": 100,
                                                              "speculative_num_steps": {"private": "PRIVATE_SENTINEL"},
                                                              "memory_limit_bytes": float("nan"), "PRIVATE_SENTINEL": 123}, "host": "PRIVATE_SENTINEL"}))
            result = compare(path, {"tp_size": 2, "generation_throughput_reported_tps": 125})
            self.assertEqual(result["fields"]["generation_throughput_reported_tps"]["change_percent"]["value"], 25)
            self.assertEqual(result["fields"]["tp_size"]["difference"]["value"], 0)
            self.assertIsNone(result["fields"]["speculative_num_steps"]["baseline"]["value"])
            self.assertIsNone(result["fields"]["memory_limit_bytes"]["baseline"]["value"])
            self.assertNotIn("PRIVATE_SENTINEL", json.dumps(result))
            path.write_text("x" * (diagnostics.MAX_BYTES + 1))
            self.assertEqual(compare(path, {})["status"]["reason"], "oversized")
        self.assertEqual(diagnostics.comparison_values({"endpoint": {"settings": {"tp_size": diagnostics.field(2)}}})["tp_size"], 2)

    def test_cli_bounds_and_errors_never_echo_sensitive_arguments(self):
        parse = self.api("parse_args")
        args = parse(["--output", "/tmp/report.json", "--gpus", "5,2", "--duration", "1", "--interval", "0.5"])
        self.assertEqual(args.gpus, [5, 2])
        self.assertEqual(args.duration, 1)
        self.assertIsNone(args.endpoint)
        self.assertIsNone(args.container)
        for options in [["--gpus", "0,0"], ["--gpus", "GPU-PRIVATE_SENTINEL"], ["--gpus", "-1"],
                        ["--duration", "61"], ["--duration", "nan"], ["--duration", "-1"],
                        ["--interval", "0"], ["--interval", "11"], ["--duration", "60", "--interval", ".25"],
                        ["--container", "-PRIVATE_SENTINEL"], ["--endpoint", "http://alice:PRIVATE_SENTINEL@localhost"],
                        ["--token-env", "PRIVATE_SENTINEL"], ["--token", "PRIVATE_SENTINEL"]]:
            with patch("sys.stderr", new_callable=io.StringIO) as stderr, self.assertRaises(SystemExit) as error:
                parse(["--output", "/tmp/PRIVATE_SENTINEL.json", *options])
            self.assertEqual(error.exception.code, 2)
            self.assertNotIn("PRIVATE_SENTINEL", stderr.getvalue())

    def test_sampler_executes_existing_load_reads_and_records_actual_intervals(self):
        build = self.api("build_report")
        args = diagnostics.parse_args(["--output", "/tmp/unused.json", "--gpus", "3", "--duration", "0.5", "--interval", "0.25"])
        clock = [0.0]

        def sleep(seconds):
            clock[0] += seconds

        def command(argv, **kwargs):
            self.assertEqual(argv[0], "nvidia-smi")
            return diagnostics.Raw(reason="tool_missing")

        with patch.object(diagnostics.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(diagnostics.time, "sleep", side_effect=sleep), \
                patch.object(diagnostics, "run_bounded", side_effect=command):
            report = build(args, None)
        self.assertEqual(len(report["samples"]), 3)
        self.assertEqual(len(report["intervals"]), 2)
        self.assertEqual(report["intervals"][0]["seconds"]["value"], .25)
        self.assertEqual(report["sampling"]["actual_window_seconds"]["value"], .5)
        for sample in report["samples"]:
            self.assertEqual(list(sample["gpu"]["devices"]), ["GPU0"])
            self.assertNotIn("GPU3", json.dumps(sample))

    def test_default_cli_writes_private_reports_without_network_or_docker(self):
        main = self.api("main")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "PRIVATE_SENTINEL.json"
            with patch.object(diagnostics, "run_bounded", return_value=diagnostics.Raw(reason="tool_missing")) as commands, \
                    patch.object(diagnostics, "collect_endpoint", side_effect=AssertionError("network forbidden")), \
                    patch.object(diagnostics, "collect_container", side_effect=AssertionError("container forbidden")), \
                    patch("sys.stdout", new_callable=io.StringIO) as stdout:
                self.assertEqual(main(["--output", str(output)]), 0)
            report = json.loads(output.read_text())
            readable = Path(str(output) + ".txt").read_text()
            self.assertEqual(report["schema_version"], 1)
            self.assertEqual(len(report["samples"]), 1)
            self.assertFalse(report["intervals"])
            self.assertEqual(report["gpu_discovery"]["reason"], "tool_missing")
            self.assertEqual(report["endpoint"]["reason"], "not_opted_in")
            self.assertTrue(all(call.args[0][0] == "nvidia-smi" for call in commands.call_args_list))
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            for phrase in ["activity", "saturation", "P2P", "bandwidth", "NODE", "PHB", "idle", "thermal", "decode", "fleet", "EOS", "warmup"]:
                self.assertIn(phrase, readable)
            self.assertNotIn("PRIVATE_SENTINEL", output.read_text() + readable + stdout.getvalue())
            with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                self.assertEqual(main(["--output", str(output)]), 2)
            self.assertNotIn("PRIVATE_SENTINEL", stderr.getvalue())

    def test_script_entrypoint_executes_as_a_real_local_process(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "local.json"
            result = subprocess.run([sys.executable, "-B", str(SCRIPT), "--output", str(output), "--gpus", "4095"],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output.read_text())["schema_version"], 1)

    def test_container_cgroup_counters_are_sampled_without_repeated_exec(self):
        args = diagnostics.parse_args(["--output", "/tmp/unused.json", "--gpus", "0", "--container", "safe", "--duration", ".25", "--interval", ".25"])
        clock = [0.0]
        calls = []

        def command(argv, **kwargs):
            calls.append(argv)
            if "inspect" in argv:
                return diagnostics.Raw(json.dumps([{"Image": "sha256:" + "b" * 64, "State": {"Pid": 1234}}]))
            return diagnostics.Raw(reason="unsupported")

        def cgroup(*args, **kwargs):
            return {"cpu_counters": {"throttled_us": diagnostics.field(100 + 80 * clock[0])}}

        with patch.object(diagnostics.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(diagnostics.time, "sleep", side_effect=lambda delay: clock.__setitem__(0, clock[0] + delay)), \
                patch.object(diagnostics, "run_bounded", side_effect=command), \
                patch.object(diagnostics, "docker_pid_namespace_verified", return_value=True), \
                patch.object(diagnostics, "collect_cgroup", side_effect=cgroup) as reads:
            report = diagnostics.build_report(args, None)
        self.assertEqual(report["intervals"][0]["cgroup_deltas"]["container_cgroup"]["cpu"]["throttled_us"]["value"], 20)
        self.assertEqual(sum("exec" in call for call in calls), 1)
        self.assertTrue(any(call.kwargs.get("subject") == "1234" for call in reads.call_args_list))
        self.assertNotIn("1234", json.dumps(report))

    def test_server_info_top_level_settings_are_allowlisted(self):
        source = {"tp_size": 2, "speculative_num_steps": 3, "model_path": "PRIVATE_SENTINEL"}
        result = diagnostics.parse_server(source)
        self.assertEqual(result["settings"]["tp_size"]["value"], 2)
        self.assertEqual(result["settings"]["speculative_num_steps"]["value"], 3)
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(result))
        source["server_args"] = {"tp_size": 4}
        self.assertEqual(diagnostics.parse_server(source)["settings"]["tp_size"]["value"], 4)

    def test_current_driver_ansi_topology_and_singular_event_tags(self):
        raw = diagnostics.Raw('\t\x1b[4mGPU0\tGPU1\x1b[0m\nGPU0 X NODE\nGPU1 NODE X\n')
        self.assertEqual(diagnostics.parse_matrix(raw, [0, 1])["GPU0"]["GPU1"]["value"], "NODE")
        xml = '<nvidia_smi_log><gpu><clocks_event_reasons><clocks_event_reason_sw_power_cap>Not Active</clocks_event_reason_sw_power_cap></clocks_event_reasons><clocks_event_reasons_counters><clocks_event_reasons_counters_sw_therm_slowdown>123 us</clocks_event_reasons_counters_sw_therm_slowdown></clocks_event_reasons_counters></gpu></nvidia_smi_log>'
        result = diagnostics.parse_gpu(diagnostics.Raw(xml))
        self.assertIs(result['clock_events']['sw_power_cap']['value'], False)
        self.assertEqual(result['event_counters_us']['sw_thermal_slowdown']['value'], 123)

    def test_v2_root_missing_limit_files_do_not_hide_child_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "child"
            child.mkdir()
            (child / "cpu.max").write_text("150000 100000")
            (child / "memory.max").write_text("1024")
            self.assertEqual(diagnostics.hierarchy_limit((child, root), "cpu", True)["value"], 1.5)
            self.assertEqual(diagnostics.hierarchy_limit((child, root), "memory", True)["value"], 1024)
            (child / "memory.max").unlink()
            self.assertIsNone(diagnostics.hierarchy_limit((child, root), "memory", True)["value"])


if __name__ == "__main__":
    unittest.main()

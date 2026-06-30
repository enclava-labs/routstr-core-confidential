#!/usr/bin/env python3
"""Generate a best-effort CAP-compatible wrapper around an OCI image.

The wrapper is intentionally conservative: it preserves the source image and
builds a new image that CAP can launch through a stable argv target. It cannot
make applications safe if they require root, write to hard-coded read-only
rootfs paths, or cannot be configured to serve the selected port.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CAP_WRAP_GO = r'''
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"syscall"
	"time"
)

func envDefault(name string, value string) string {
	if current := os.Getenv(name); current != "" {
		return current
	}
	_ = os.Setenv(name, value)
	return value
}

func ensureRuntimeDirs() {
	dataDir := envDefault("ROUTSTR_DATA_DIR", "/app/data")
	logDir := envDefault("ROUTSTR_LOG_DIR", "/run/enclava/routstr-logs")
	envDefault("HOME", dataDir)
	envDefault("TMPDIR", dataDir+"/tmp")
	envDefault("XDG_CACHE_HOME", dataDir+"/.cache")
	envDefault("XDG_CONFIG_HOME", dataDir+"/.config")
	envDefault("XDG_DATA_HOME", dataDir+"/.local/share")

	for _, dir := range []string{
		dataDir,
		logDir,
		os.Getenv("TMPDIR"),
		os.Getenv("XDG_CACHE_HOME"),
		os.Getenv("XDG_CONFIG_HOME"),
		os.Getenv("XDG_DATA_HOME"),
	} {
		if err := os.MkdirAll(dir, 0o770); err != nil {
			log.Fatalf("cap-wrap: create runtime dir %s: %v", dir, err)
		}
	}
}

func loadCommandFromFile(path string) []string {
	raw, err := os.ReadFile(path)
	if err != nil {
		log.Fatalf("cap-wrap: read original argv file %s: %v", path, err)
	}
	var argv []string
	if err := json.Unmarshal(raw, &argv); err != nil {
		log.Fatalf("cap-wrap: parse original argv file %s: %v", path, err)
	}
	return argv
}

func resolveCommand(argv []string) []string {
	if len(argv) > 0 && argv[0] == "--" {
		argv = argv[1:]
	}
	if len(argv) > 0 {
		return argv
	}
	if raw := os.Getenv("CAP_ORIGINAL_ARGV_JSON"); raw != "" {
		var decoded []string
		if err := json.Unmarshal([]byte(raw), &decoded); err != nil {
			log.Fatalf("cap-wrap: parse CAP_ORIGINAL_ARGV_JSON: %v", err)
		}
		return decoded
	}
	if path := os.Getenv("CAP_ORIGINAL_ARGV_FILE"); path != "" {
		return loadCommandFromFile(path)
	}
	log.Fatalf("cap-wrap: no command configured")
	return nil
}

func envWithPort(port string) []string {
	env := os.Environ()
	found := false
	for i, entry := range env {
		if len(entry) >= 5 && entry[:5] == "PORT=" {
			env[i] = "PORT=" + port
			found = true
			break
		}
	}
	if !found {
		env = append(env, "PORT="+port)
	}
	return env
}

func exitFromWaitErr(err error) {
	if err == nil {
		os.Exit(0)
	}
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		if status, ok := exitErr.Sys().(syscall.WaitStatus); ok {
			if status.Signaled() {
				os.Exit(128 + int(status.Signal()))
			}
			os.Exit(status.ExitStatus())
		}
	}
	log.Printf("cap-wrap: child exited with error: %v", err)
	os.Exit(1)
}

func runDirect(argv []string) {
	if err := syscall.Exec(argv[0], argv, os.Environ()); err != nil {
		log.Fatalf("cap-wrap: exec %s: %v", argv[0], err)
	}
}

func runProxy(argv []string, capPort string, upstreamPort string, healthPath string) {
	target, err := url.Parse("http://127.0.0.1:" + upstreamPort)
	if err != nil {
		log.Fatalf("cap-wrap: invalid upstream port %q: %v", upstreamPort, err)
	}
	proxy := httputil.NewSingleHostReverseProxy(target)
	server := &http.Server{
		Addr:              "0.0.0.0:" + capPort,
		ReadHeaderTimeout: 5 * time.Second,
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == healthPath {
				w.Header().Set("Content-Type", "text/plain; charset=utf-8")
				_, _ = w.Write([]byte("ok\n"))
				return
			}
			proxy.ServeHTTP(w, r)
		}),
	}

	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Env = envWithPort(upstreamPort)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin
	if err := cmd.Start(); err != nil {
		log.Fatalf("cap-wrap: start child %s: %v", argv[0], err)
	}

	sigCh := make(chan os.Signal, 2)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		for sig := range sigCh {
			if cmd.Process != nil {
				_ = cmd.Process.Signal(sig)
			}
		}
	}()

	serverErr := make(chan error, 1)
	go func() {
		err := server.ListenAndServe()
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			serverErr <- err
			return
		}
		serverErr <- nil
	}()

	waitErr := make(chan error, 1)
	go func() {
		waitErr <- cmd.Wait()
	}()

	select {
	case err := <-serverErr:
		if err != nil {
			_ = cmd.Process.Signal(syscall.SIGTERM)
			log.Fatalf("cap-wrap: proxy failed: %v", err)
		}
		exitFromWaitErr(<-waitErr)
	case err := <-waitErr:
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(ctx)
		exitFromWaitErr(err)
	}
}

func main() {
	log.SetFlags(0)
	ensureRuntimeDirs()

	capPort := envDefault("PORT", "8000")
	capPort = envDefault("CAP_PORT", capPort)
	mode := envDefault("CAP_MODE", "direct")
	upstreamPort := envDefault("CAP_UPSTREAM_PORT", fmt.Sprintf("%d", mustAtoi(capPort)+1))
	healthPath := envDefault("CAP_HEALTH_PATH", "/health")
	argv := resolveCommand(os.Args[1:])
	if len(argv) == 0 || argv[0] == "" {
		log.Fatalf("cap-wrap: empty command")
	}

	if mode == "proxy" {
		runProxy(argv, capPort, upstreamPort, healthPath)
		return
	}
	if mode != "direct" {
		log.Fatalf("cap-wrap: unsupported CAP_MODE %q", mode)
	}
	runDirect(argv)
}

func mustAtoi(raw string) int {
	var value int
	_, err := fmt.Sscanf(raw, "%d", &value)
	if err != nil || value <= 0 {
		return 8000
	}
	return value
}
'''


@dataclass
class WrapperConfig:
    source_image: str
    target_image: str
    command: list[str]
    output_dir: Path
    app_name: str
    port: int = 8000
    mode: str = "direct"
    upstream_port: int | None = None
    health_path: str = "/health"
    storage_paths: list[str] = field(default_factory=lambda: ["/app/data"])
    data_dir: str = "/app/data"
    log_dir: str = "/run/enclava/routstr-logs"
    build: bool = True
    runtime: str = "docker"
    platform: str | None = None
    pull: bool = False


def _run_json(command: list[str]) -> Any:
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def runtime_command(runtime: str) -> list[str]:
    return shlex.split(runtime)


def inspect_image_command(runtime: str, source_image: str, *, pull: bool = False) -> list[str]:
    runtime_argv = runtime_command(runtime)
    try:
        inspected = _run_json([*runtime_argv, "image", "inspect", source_image])
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        if not pull:
            raise
        subprocess.run([*runtime_argv, "pull", source_image], check=True)
        inspected = _run_json([*runtime_argv, "image", "inspect", source_image])

    if not inspected:
        raise ValueError(f"no image metadata returned for {source_image}")
    config = inspected[0].get("Config") or {}
    entrypoint = config.get("Entrypoint") or []
    cmd = config.get("Cmd") or []
    if isinstance(entrypoint, str):
        entrypoint = [entrypoint]
    if isinstance(cmd, str):
        cmd = [cmd]
    argv = [str(arg) for arg in [*entrypoint, *cmd] if str(arg)]
    if not argv:
        raise ValueError(
            f"{source_image} has no default Entrypoint/Cmd; pass --command-json"
        )
    return argv


def parse_command_json(raw: str) -> list[str]:
    decoded = json.loads(raw)
    if not isinstance(decoded, list) or not all(isinstance(item, str) for item in decoded):
        raise ValueError("--command-json must be a JSON string array")
    if not decoded or not decoded[0]:
        raise ValueError("--command-json cannot be empty")
    return decoded


def default_app_name(source_image: str) -> str:
    image = source_image.rsplit("/", 1)[-1].split("@", 1)[0].split(":", 1)[0]
    slug = re.sub(r"[^a-z0-9-]+", "-", image.lower()).strip("-")
    return slug or "cap-app"


def toml_string(value: str) -> str:
    return json.dumps(value)


def toml_string_array(values: list[str]) -> str:
    return "[" + ", ".join(toml_string(value) for value in values) + "]"


def render_dockerfile(config: WrapperConfig) -> str:
    upstream = config.upstream_port or config.port + 1
    return textwrap.dedent(
        f"""\
        # syntax=docker/dockerfile:1
        ARG SOURCE_IMAGE=scratch
        ARG GO_BUILDER_IMAGE=golang:1.24-alpine

        FROM --platform=$BUILDPLATFORM ${{GO_BUILDER_IMAGE}} AS cap-helper
        ARG TARGETOS=linux
        ARG TARGETARCH=amd64
        WORKDIR /src
        COPY cap-wrap.go .
        RUN CGO_ENABLED=0 GOOS="${{TARGETOS}}" GOARCH="${{TARGETARCH}}" go build -trimpath -ldflags="-s -w" -o /out/cap-wrap ./cap-wrap.go \\
            && mkdir -p \\
                /cap-root/usr/local/bin \\
                /cap-root/usr/local/share \\
                /cap-root/run/enclava \\
                /cap-root{config.data_dir}/tmp \\
                /cap-root{config.data_dir}/.cache \\
                /cap-root{config.data_dir}/.config \\
                /cap-root{config.data_dir}/.local/share \\
                /cap-root{config.log_dir} \\
            && cp /out/cap-wrap /cap-root/usr/local/bin/cap-wrap \\
            && chmod 0555 /cap-root/usr/local/bin/cap-wrap \\
            && chown -R 10001:10001 /cap-root{config.data_dir} /cap-root{config.log_dir} /cap-root/run/enclava

        FROM ${{SOURCE_IMAGE}}
        COPY --from=cap-helper /cap-root/ /
        COPY cap-command.json /usr/local/share/cap-command.json

        ENV CAP_ORIGINAL_ARGV_FILE=/usr/local/share/cap-command.json
        ENV CAP_MODE={config.mode}
        ENV CAP_PORT={config.port}
        ENV CAP_UPSTREAM_PORT={upstream}
        ENV CAP_HEALTH_PATH={config.health_path}
        ENV PORT={config.port}
        ENV ROUTSTR_DATA_DIR={config.data_dir}
        ENV ROUTSTR_LOG_DIR={config.log_dir}
        ENV HOME={config.data_dir}
        ENV TMPDIR={config.data_dir}/tmp
        ENV XDG_CACHE_HOME={config.data_dir}/.cache
        ENV XDG_CONFIG_HOME={config.data_dir}/.config
        ENV XDG_DATA_HOME={config.data_dir}/.local/share

        USER 10001:10001
        EXPOSE {config.port}
        ENTRYPOINT ["/usr/local/bin/cap-wrap"]
        CMD []
        """
    )


def render_enclava_toml(config: WrapperConfig) -> str:
    return textwrap.dedent(
        f"""\
        [app]
        name = {toml_string(config.app_name)}
        port = {config.port}
        command = ["/usr/local/bin/cap-wrap"]

        [storage]
        paths = {toml_string_array(config.storage_paths)}
        size = "10Gi"
        tls_size = "2Gi"

        [unlock]
        mode = "auto"

        [egress]
        allow = []

        [resources]
        cpu = "1"
        memory = "1Gi"

        [health]
        path = {toml_string(config.health_path)}
        interval = 30
        timeout = 5
        """
    )


def render_readme(config: WrapperConfig) -> str:
    command = " ".join(shlex.quote(part) for part in config.command)
    mode_note = (
        "The wrapper serves /health itself and proxies requests to the child app. "
        "The child receives PORT set to CAP_UPSTREAM_PORT."
        if config.mode == "proxy"
        else "The wrapped app must serve the configured health path itself."
    )
    return textwrap.dedent(
        f"""\
        # CAP Wrapper Image

        Source image: `{config.source_image}`
        Target image: `{config.target_image}`
        Wrapped command: `{command}`
        Mode: `{config.mode}`

        Use `enclava.toml` from this directory or copy its `[app]` command:

        ```toml
        command = ["/usr/local/bin/cap-wrap"]
        ```

        {mode_note}

        Limits: this wrapper cannot fix apps that require root, write outside
        CAP writable mounts under a read-only root filesystem, bind only to an
        unavailable port, or need extra Linux capabilities.
        """
    )


def write_build_context(config: WrapperConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "Dockerfile").write_text(render_dockerfile(config))
    (config.output_dir / "cap-wrap.go").write_text(CAP_WRAP_GO.lstrip())
    (config.output_dir / "cap-command.json").write_text(
        json.dumps(config.command, indent=2) + "\n"
    )
    (config.output_dir / "enclava.toml").write_text(render_enclava_toml(config))
    (config.output_dir / "README.cap-wrapper.md").write_text(render_readme(config))
    (config.output_dir / ".dockerignore").write_text(
        "*\n!Dockerfile\n!cap-wrap.go\n!cap-command.json\n"
    )


def build_image(config: WrapperConfig) -> None:
    command = [
        *runtime_command(config.runtime),
        "build",
        "-t",
        config.target_image,
        "--build-arg",
        f"SOURCE_IMAGE={config.source_image}",
    ]
    if config.platform:
        command.extend(["--platform", config.platform])
    command.append(str(config.output_dir))
    subprocess.run(command, check=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and optionally build a CAP-compatible wrapper image."
    )
    parser.add_argument("source_image", help="Existing image to wrap")
    parser.add_argument("target_image", help="Tag for the generated wrapper image")
    parser.add_argument(
        "--command-json",
        help="Override source Entrypoint/Cmd with a JSON argv array",
    )
    parser.add_argument(
        "--mode",
        choices=("direct", "proxy"),
        default="direct",
        help=(
            "direct execs the app and requires it to serve /health; "
            "proxy injects /health and forwards traffic to CAP_UPSTREAM_PORT"
        ),
    )
    parser.add_argument("--port", type=int, default=8000, help="CAP app port")
    parser.add_argument(
        "--upstream-port",
        type=int,
        help="Child app port when --mode proxy is used; defaults to port + 1",
    )
    parser.add_argument("--health-path", default="/health")
    parser.add_argument("--app-name", help="App name for generated enclava.toml")
    parser.add_argument(
        "--storage-path",
        action="append",
        dest="storage_paths",
        help="CAP writable storage path; repeat for multiple paths",
    )
    parser.add_argument("--data-dir", default="/app/data")
    parser.add_argument("--log-dir", default="/run/enclava/routstr-logs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for generated Dockerfile/enclava.toml",
    )
    parser.add_argument(
        "--runtime",
        default="docker",
        help='container runtime command, for example "docker", "podman", or "sudo docker"',
    )
    parser.add_argument("--platform", help="Optional docker build --platform")
    parser.add_argument("--pull", action="store_true", help="Pull source image if needed")
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Only write the generated context; do not build the target image",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> WrapperConfig:
    command = (
        parse_command_json(args.command_json)
        if args.command_json
        else inspect_image_command(args.runtime, args.source_image, pull=args.pull)
    )
    app_name = args.app_name or default_app_name(args.source_image)
    output_dir = args.output_dir or Path("build") / "cap-wrapper" / app_name
    storage_paths = args.storage_paths or [args.data_dir]
    if args.mode == "proxy" and args.upstream_port == args.port:
        raise ValueError("--upstream-port must differ from --port in proxy mode")
    return WrapperConfig(
        source_image=args.source_image,
        target_image=args.target_image,
        command=command,
        output_dir=output_dir,
        app_name=app_name,
        port=args.port,
        mode=args.mode,
        upstream_port=args.upstream_port,
        health_path=args.health_path,
        storage_paths=storage_paths,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        build=not args.no_build,
        runtime=args.runtime,
        platform=args.platform,
        pull=args.pull,
    )


def main(argv: list[str]) -> int:
    try:
        config = config_from_args(parse_args(argv))
        write_build_context(config)
        print(f"Wrote CAP wrapper context: {config.output_dir}", flush=True)
        print('Generated CAP app command: ["/usr/local/bin/cap-wrap"]', flush=True)
        if config.build:
            build_image(config)
            print(f"Built wrapper image: {config.target_image}")
        else:
            print("Skipped image build (--no-build).")
        return 0
    except Exception as exc:
        print(f"cap_image_wrapper.py: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

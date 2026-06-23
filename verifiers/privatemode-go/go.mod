module github.com/edgelesssys/continuum/routstr-privatemode-verifier

go 1.26.0

toolchain go1.26.4

require (
	github.com/edgelesssys/continuum v1.42.0
	github.com/edgelesssys/contrast v1.20.0
)

require (
	github.com/beorn7/perks v1.0.1 // indirect
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/evanphx/json-patch/v5 v5.9.11 // indirect
	github.com/google/go-sev-guest v0.14.2-0.20251119154202-af1c107a648f // indirect
	github.com/google/go-tdx-guest v0.3.2-0.20260104162950-32866d7a678f // indirect
	github.com/google/logger v1.1.2 // indirect
	github.com/google/uuid v1.6.0 // indirect
	github.com/mdlayher/socket v0.5.1 // indirect
	github.com/mdlayher/vsock v1.2.1 // indirect
	github.com/munnerz/goautoneg v0.0.0-20191010083416-a7dc8b61c822 // indirect
	github.com/prometheus/client_golang v1.23.2 // indirect
	github.com/prometheus/client_model v0.6.2 // indirect
	github.com/prometheus/common v0.67.5 // indirect
	github.com/prometheus/procfs v0.20.1 // indirect
	github.com/spf13/afero v1.15.0 // indirect
	go.uber.org/multierr v1.11.0 // indirect
	go.yaml.in/yaml/v2 v2.4.3 // indirect
	golang.org/x/crypto v0.52.0 // indirect
	golang.org/x/exp v0.0.0-20260410095643-746e56fc9e2f // indirect
	golang.org/x/net v0.54.0 // indirect
	golang.org/x/sync v0.20.0 // indirect
	golang.org/x/sys v0.45.0 // indirect
	golang.org/x/text v0.37.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20260511170946-3700d4141b60 // indirect
	google.golang.org/grpc v1.81.1 // indirect
	google.golang.org/protobuf v1.36.12-0.20260120151049-f2248ac996af // indirect
	k8s.io/utils v0.0.0-20260507154919-ff6756f316d2 // indirect
)

replace github.com/edgelesssys/continuum => github.com/edgelesssys/privatemode-public v1.42.0

replace github.com/google/go-tdx-guest => github.com/edgelesssys/go-tdx-guest v0.0.0-20260120144929-ac8c4b4a23eb

replace github.com/russellhaering/goxmldsig => github.com/daniel-weisse/goxmldsig v0.0.0-20250722142009-1e8a5af35ad3

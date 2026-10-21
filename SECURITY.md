# Security

Please report vulnerabilities privately to <contact@inco.ai> rather than in a
public issue. Include the Splash version (`splash --version`), macOS version,
and steps to reproduce. We will acknowledge your report and keep you informed
until it is resolved.

Splash serves on `127.0.0.1`, and authentication is off by default. Before
exposing the server beyond the local machine, set a key with `SPLASH_API_KEY`
or `splash serve --api-key`; API requests then need it as a bearer token or
`x-api-key`. `--allowed-origin '*'` lets every web page open in a browser that
reaches the server use it, so set a key with it. Crash traces, which Splash
records only with `SPLASH_CRASH_TRACE=1`, can contain private conversation
data. Exposing the server without a key or a proxy is outside the threat model.

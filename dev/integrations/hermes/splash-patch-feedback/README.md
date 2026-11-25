# Hermes patch feedback

This opt-in plug-in attaches the result of the existing, zero-argument
`run_visible_tests` file tool to a successful `patch` in an isolated checkout.
A failed edit or an unavailable checker retains the normal tool result.
Use only with a trusted local check already approved for that checkout.
The plug-in accepts neither shell commands nor paths from a model to select
the test suite.

Install the directory in an isolated Hermes profile's `plugins/` directory.
Enable `splash-patch-feedback` in that profile and set
`plugins.entries.splash-patch-feedback.settings.enabled` to true. Set
`settings.endpoint` to the profile's loopback OpenAI endpoint ending in `/v1`,
`settings.route_model` to its precise model alias, and
`settings.workspace` to the absolute path to the isolated checkout.
Include the approved check in the profile's visible file tools. Instruct the
agent to use the attached `visible_tests` result before making another edit.
The existing test tool can still be called separately.

The changes do not alter the tool schema sent to Splash or replace final
acceptance checks.

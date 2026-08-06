# Lazy Project Directories

## Goal

Quasar must not create `media`, `models`, or `screenshots` directories merely because the library is imported, initialized, or asked to resolve paths. These directories are created only immediately before a feature writes data into them.

## Compatibility

The public path functions continue to return `Path` values and keep the existing resolver and environment-variable precedence. No function signatures or configuration names change. Hosts that only use text, runtime, agent, or tool-registry capabilities therefore stop receiving unused directories without requiring Quasar-specific feature flags.

## Design

`get_project_media_dir()`, `get_project_models_dir()`, and `get_project_screenshots_dir()` become pure path resolvers. `get_default_paths()` and startup path logging may call them safely because resolution has no filesystem side effect.

Directory creation moves to concrete filesystem writers. The local media adapter creates its resolved media directory immediately before opening an output file. Model and screenshot features must apply the same rule at their write boundaries; existing model writers that already create their output directory retain that behavior. A path getter must never be used as an implicit directory initializer.

Resolver-provided and environment-provided paths follow the same lazy rule. If directory creation fails at an actual write boundary, the write operation raises its normal filesystem error instead of silently treating path resolution as successful.

## Tests

- Resolving each default path leaves the filesystem unchanged.
- `get_default_paths()` leaves all optional project directories absent.
- Saving local media creates `media` and writes the requested file.
- Environment and host resolver overrides remain unchanged and are not eagerly created.
- Existing Quasar tests continue to pass.

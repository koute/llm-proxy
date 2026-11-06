# Findings of the A/B hunt against LiteLLM, and their tests

Every bug the A/B hunt (`ab.py`, five hunters, rounds 1–5) found in the gateway, with
the test which guards its fix. Use it when making the diff smaller: a refactoring
which keeps these tests green keeps the fixes. `cargo test` runs all of them.

Tests are in `tests/gateway.rs` (G), `tests/middle_end.rs` (M), `tests/sigusr1.rs`
(S), `tests/logging.rs` (L) and unit tests (`path::to::test`).

## Keys, auth and the admin API

| Finding | Test |
|---|---|
| Concurrent `create-key` and admin writes lose keys (no cross-process lock) | `gateway::tests::concurrent_changes_keep_every_key` |
| A change which timed out (hung mount) was still applied later | `gateway::tests::abandoned_changes_are_not_written` |
| A hung change kept the keys lock forever | `gateway::tests::abandoned_changes_release_the_lock` |
| One hung access blocked every later one; then probes, recovery | `gateway::tests::hung_accesses_and_recovery` |
| Lock file of `sudo create-key` (strict umask) unusable by the middle end | `gateway::tests::lock_files_are_readable_for_everyone` |
| `create-key` followed symlinks planted as lock file / audit log | `gateway::tests::lock_files_are_not_followed`, `gateway::audit::tests::appends_never_follow_links` |
| Symlinked keys file replaced by a regular file | `gateway::tests::changes_follow_symbolic_links` |
| Invalid keys file (duplicate ids) answered 400, not counted as file failure | `gateway::tests::an_invalid_file_is_a_file_failure` |
| Invalid keys file kept `serve` from starting | `config::tests::invalid_keys_file` |
| Hung keys file at startup blocked forever | `config::tests::hung_keys_file_at_startup` |
| `create-key` created a new file when the mount was down; dangling links | G `create_key_command` |
| Plaintext unknown API keys in the log | L `rejection_log` |
| Last admin locked out via a near expiry | G `admin_api_guards` |
| Lasting-admin rule blocked unrelated changes; needs the whole admin API | G `round_two_key_rules` |
| Route patterns which never match accepted; routes not enforced on /admin, /metrics | G `admin_api_guards` |
| Routes naming the `/v1`-less path never matched | G `round_two_key_rules` |
| `api-key` header, empty `Authorization`, LiteLLM's header precedence | G `admin_api_guards`, G `round_two_key_rules` |
| Foreign credential headers (`x-goog-api-key`, ...) forwarded to the engine | G `round_two_key_rules` |
| Blocked keys 403 instead of LiteLLM's 401 | G `keys_authenticate_and_are_audited`, G `admin_api_manages_keys` |
| Account keys audited as unknown on /metrics, /admin | G `aborted_requests_keep_their_status_and_first_token_time` |
| /admin could be framed | G `admin_api_guards` |
| `keys_file_errors_total` counted admin mistakes | G `admin_api_guards` |
| Hand edits picked up by an admin write not audited | G `hand_edits_picked_up_by_admin_changes_are_audited` |
| Restart served the pre-flight's stale keys | G `restarts_keep_the_current_keys` |

## Limits

| Finding | Test |
|---|---|
| tpm/tpd overshoot by concurrent requests (no reservations) | `gateway::limits::tests::reservations_count_until_requests_finish` |
| Retry-After too short; several exceeded limits; limit 0; overflow | `gateway::limits::tests::retry_after_is_enough`, `rate_limits` |
| Reservations lost across SIGUSR1 | G `reservations_survive_restarts` |
| Reservations invisible in usage and metrics | G `reservations_show_in_usage_and_metrics` |
| queue_full consumed rpm, no Retry-After | G `full_queues_refund_the_key` |
| Passthrough requests bypassed limits and audit | G `passthrough_requests_count_against_limits_and_are_audited` |
| Estimate: JSON overhead, images, max_completion_tokens 0, no max_tokens, smallest limit, fake data URLs, tools and tool calls like LiteLLM | `middle_end::routing::estimate_tests::estimated_tokens_like_litellm` |
| Too-large requests reported as "remaining 0" | `gateway::limits::tests::retry_after_is_enough` |

## Responses, paths and HTTP

| Finding | Test |
|---|---|
| Early 200 head hid engine errors of non-streaming requests | M `queued_non_streaming_requests_get_the_engine_status` |
| Late engine errors in streams invisible to SDKs (OpenAI, Anthropic, 2xx JSON) | M `late_engine_errors_of_streams_are_error_events`, M `late_engine_errors_of_anthropic_streams`, M `late_responses_which_are_no_event_stream_are_error_events`, `middle_end::request_flow::sse_error_tests` |
| Engine unreachable after the early head cut the stream | M `single_response_head_after_unreachable_engine` |
| Closed clients held their slot until the next write (also with buffered bytes) | G `closed_clients_release_their_requests_at_once` |
| 413 became 502 behind nginx (no lingering close) | M `too_large_streamed_bodies_get_a_413` |
| Slow uploads 408 while bytes flowed; broken bodies 502 | M `broken_and_slow_bodies` |
| 1-byte chunks: quadratic CPU | M `one_byte_chunks_arrive_intact` |
| More than ~100 headers rejected | M `many_request_headers` |
| CORS | M `cors_like_litellm` |
| Non-canonical paths admitted; `/v1`-less paths; 405; probes; bounded path labels | G `paths_like_litellm` |
| SIGUSR1 could move the middle end socket away from the frontend | S `restarts_refuse_a_moved_middle_end_socket` |

## Audit log, metrics and restarts

| Finding | Test |
|---|---|
| Audit log off for good when it can't be opened | `utils::tests::unwritable_records_are_handed_back`, `records_are_written_once_the_file_works` |
| Unwritten audit / scheduling log events lost at SIGUSR1 | G `unwritten_audit_events_survive_restarts`, G `unwritten_scheduling_log_events_survive_restarts` |
| A failing or stuck log held up restarts | `utils::tests::a_stuck_file_does_not_hold_up_a_stop` |
| ... but a slow write which makes progress was handed over and duplicated | `utils::tests::slow_writes_which_make_progress_are_waited_for` |
| Partial lines and duplicates after a full disk | `utils::tests::retries_start_at_the_record_written_in_part`, `reopen_ends_a_partial_line` |
| Aborted requests without status/TTFT; engine failures recorded as aborts | G `aborted_requests_keep_their_status_and_first_token_time`, G `engine_errors_and_failures_in_the_audit_log` |
| Engine-rejected requests recorded 5 s late | G `engine_errors_and_failures_in_the_audit_log` |
| Class and times lost across SIGUSR1 (TTFT > total) | G `restarts_keep_running_requests_attributed_and_limits` |
| Generating time lost across SIGUSR1 | G `generating_time_survives_restarts` |
| Queue order lost across SIGUSR1 | G `restarts_keep_the_queue_in_order` |
| Passthrough requests across SIGUSR1 unrecorded | G `passthrough_requests_survive_restarts` |
| Queued stream left after the early head: status null | G `queued_streams_left_after_the_early_head_are_aborted_with_200` |
| Path, user agent, model, client not cut | G `long_fields_are_cut_in_the_audit_log` |
| Unbounded virtual users | G `virtual_users_are_cut` |
| The proxy's own 405/408/413 not counted | G `the_proxy_s_own_rejections_are_counted` |
| Gauges double-counted and vanished when idle | `gateway::metrics::tests::exposition_format`, `quality_of_service::tests::status_counts` |

## Not covered by automated tests

These fixes are verified only by the hunt; each needs an environment the tests don't
have.

- A request the engine finished just before SIGUSR1 audited as `unknown`: a race
  between the engine's event and the response, seen only under heavy load
  (`take_finished_record`).
- Truncating the log while the disk is full (needs a full disk); the length check in
  `write_batch`.
- RLIMIT_NOFILE raised at start (`raise_open_files_limit`): process-wide.
- SIGTERM doesn't wait for hung keys-file threads (`shutdown_background`): needs a hung
  file system in a separate process.
- The handover written buffered (performance).
- The admin panel's JavaScript (limit 0, `[]` routes, reserved tokens): no browser in
  the tests.
- FIFOs as log files, and a restart cutting off a write which a hung file system
  delivers in part: up to 4 KiB of events can be written twice (documented).
- `sudo create-key` replacing a service-owned keys file keeps its owner (`fchown`):
  needs root.

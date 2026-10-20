mod support;

use std::time::Duration;

use llm_proxy::benchmark::{self, BenchmarkOptions};
use llm_proxy::config::{rank_capacities, BackendConfig, QualityOfServiceConfig, Seconds};
use llm_proxy::engine_protocol::EngineKind;

use support::mock_engine::{prompt_tokens, MockEngine, TOKEN_INTERVAL};

fn options(engine: &MockEngine, directory: &tempfile::TempDir, limit: usize) -> BenchmarkOptions {
    BenchmarkOptions {
        upstream: format!("http://{}", engine.address).try_into().unwrap(),
        control_socket: engine.control_socket.to_string_lossy().into_owned(),
        output: directory.path().join("backend.json"),
        general: directory.path().join("general.json"),
        model: None,
        prompt_tokens: 200,
        output_tokens: 30,
        max_parallel_requests_per_rank: Some(limit),
        level_duration: Seconds::new(2),
    }
}

#[tokio::test]
async fn backend_file_from_measurements() {
    let directory = tempfile::tempdir().unwrap();
    let engine = MockEngine::start(directory.path().join("engine.sock")).await;
    let backend = benchmark::run(options(&engine, &directory, 4))
        .await
        .unwrap();
    let written = BackendConfig::parse(
        &std::fs::read_to_string(directory.path().join("backend.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(written.engine, EngineKind::Vllm);
    assert_eq!(written.ranks, backend.ranks);
    assert_eq!(written.model, "mock-model");
    let result = written.benchmark.unwrap();
    assert!(result.prompt_tokens.abs_diff(200) <= 10);
    let last_request = engine.received().pop().unwrap();
    assert!(prompt_tokens(&last_request.body).abs_diff(200) <= 10);
    let expected_rate = 1.0 / TOKEN_INTERVAL.as_secs_f64();
    assert!((result.single_stream_tokens_per_second - expected_rate).abs() < expected_rate * 0.3);
    assert_eq!(
        result.single_stream_tokens_per_second,
        result.measurements[0].tokens_per_second_per_request
    );
    let levels: Vec<usize> = result
        .measurements
        .iter()
        .map(|measurement| measurement.parallel_requests_per_rank)
        .collect();
    assert_eq!(levels, vec![1, 2, 3, 4]);
    for measurement in &result.measurements {
        let parallel_requests = measurement.parallel_requests_per_rank as f64;
        let aggregate = parallel_requests * expected_rate;
        assert!((measurement.tokens_per_second_per_rank - aggregate).abs() < aggregate * 0.3);
        let client_total = measurement.tokens_per_second_per_request * parallel_requests;
        assert!((client_total - measurement.tokens_per_second_per_rank).abs() < 1e-6);
    }
}

#[tokio::test]
async fn kv_cache_limit() {
    let directory = tempfile::tempdir().unwrap();
    let engine = MockEngine::start(directory.path().join("engine.sock")).await;
    engine.limit_running(3, 250);
    let backend = benchmark::run(options(&engine, &directory, 8))
        .await
        .unwrap();
    assert_eq!(backend.engine_max_parallel_requests_per_rank, 3);
    let levels: Vec<usize> = backend
        .benchmark
        .unwrap()
        .measurements
        .iter()
        .map(|measurement| measurement.parallel_requests_per_rank)
        .collect();
    assert_eq!(levels, vec![1, 2, 3]);
}

#[tokio::test]
async fn prefill_bound_throughput() {
    let directory = tempfile::tempdir().unwrap();
    let engine = MockEngine::start(directory.path().join("engine.sock")).await;
    let prefill_interval = Duration::from_millis(20);
    engine.set_prefill_interval(prefill_interval);
    let options = BenchmarkOptions {
        output_tokens: 64,
        level_duration: Seconds::try_from(0.4).unwrap(),
        ..options(&engine, &directory, 32)
    };
    let output_tokens = options.output_tokens as f64;
    let result = benchmark::run(options).await.unwrap().benchmark.unwrap();
    let levels: Vec<usize> = result
        .measurements
        .iter()
        .map(|measurement| measurement.parallel_requests_per_rank)
        .collect();
    assert_eq!(levels, vec![1, 2, 3, 4, 6, 8, 12, 16, 24, 32]);
    for measurement in &result.measurements {
        let decode_rate =
            measurement.parallel_requests_per_rank as f64 / TOKEN_INTERVAL.as_secs_f64();
        let prefill_seconds_per_token = prefill_interval.as_secs_f64() / output_tokens;
        let steady_state_rate = decode_rate / (1.0 + decode_rate * prefill_seconds_per_token);
        assert!(
            (measurement.tokens_per_second_per_rank - steady_state_rate).abs()
                < steady_state_rate * 0.3
        );
    }
}

#[tokio::test]
async fn throughput_drop_boundary() {
    let directory = tempfile::tempdir().unwrap();
    let engine = MockEngine::start(directory.path().join("engine.sock")).await;
    engine.set_large_batch_token_interval(5, TOKEN_INTERVAL * 4);
    let options = BenchmarkOptions {
        level_duration: Seconds::new(1),
        ..options(&engine, &directory, 8)
    };
    let backend = benchmark::run(options).await.unwrap();
    let levels: Vec<usize> = backend
        .benchmark
        .as_ref()
        .unwrap()
        .measurements
        .iter()
        .map(|measurement| measurement.parallel_requests_per_rank)
        .collect();
    assert_eq!(levels, vec![1, 2, 3, 4, 5, 6, 8]);
    assert_eq!(
        rank_capacities(&QualityOfServiceConfig::default(), &backend),
        Ok(vec![5])
    );
}

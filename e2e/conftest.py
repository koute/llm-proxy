ENGINES = ("vllm", "sglang")


def pytest_addoption(parser):
    parser.addoption("--engine", choices=ENGINES, help="test only this engine")


def pytest_generate_tests(metafunc):
    if "engine" in metafunc.fixturenames:
        selected = metafunc.config.getoption("--engine")
        metafunc.parametrize("engine", [selected] if selected else ENGINES, scope="module")

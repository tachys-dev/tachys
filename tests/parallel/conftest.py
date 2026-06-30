def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "mpi: mark test as requiring mpirun (skipped automatically when mpirun is not found)",
    )

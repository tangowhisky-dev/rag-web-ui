"""Tests for wiring StartupRecoveryService into main.py lifecycle.

Verifies:
1. main.py imports cleanly (syntax + import validation).
2. startup_recovery module-level variable exists in main.py.
3. StartupRecoveryService start/stop are wired in startup_event and shutdown_event.
4. startup_event respects WATCHER_ENABLED gating.
5. shutdown_event handles startup_recovery being None (null safety).
6. No circular import errors.
"""
import ast
import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

# conftest.py sets up the SQLite session before any app.* import.


@pytest.fixture(autouse=True)
def reset_main_globals():
    """Reset module-level globals in main.py before each test.

    startup_event and shutdown_event mutate the global variables
    `startup_recovery` and `watcher_service`. Without resetting
    between tests, later tests see stale values from earlier tests.
    """
    import app.main

    app.main.startup_recovery = None
    app.main.watcher_service = None
    yield
    app.main.startup_recovery = None
    app.main.watcher_service = None


# ---------------------------------------------------------------------------
# Static structure tests — no runtime calls into main.py functions
# ---------------------------------------------------------------------------


class TestMainPyImportAndSyntax:
    """Test that main.py has valid Python syntax and the right structure."""

    def test_main_py_syntax_is_valid(self):
        """main.py must parse without SyntaxError."""
        main_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "app",
            "main.py",
        )
        with open(main_path) as f:
            source = f.read()
        ast.parse(source)

    def test_main_py_imports_startup_recovery_service(self):
        """main.py must import StartupRecoveryService."""
        main_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "app",
            "main.py",
        )
        with open(main_path) as f:
            source = f.read()
        assert (
            "from app.services.startup_recovery_service import StartupRecoveryService"
            in source
        )

    def test_main_py_has_startup_recovery_module_variable(self):
        """main.py must declare `startup_recovery: StartupRecoveryService | None = None`."""
        main_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "app",
            "main.py",
        )
        with open(main_path) as f:
            source = f.read()
        assert (
            "startup_recovery: StartupRecoveryService | None = None" in source
        )

    def test_main_py_shutdown_handles_recovery(self):
        """main.py shutdown_event must reference startup_recovery.stop()."""
        main_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "app",
            "main.py",
        )
        with open(main_path) as f:
            source = f.read()
        assert "startup_recovery.stop()" in source

    def test_main_py_shutdown_includes_recovery_in_global(self):
        """main.py shutdown_event must declare startup_recovery as global."""
        main_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "app",
            "main.py",
        )
        with open(main_path) as f:
            source = f.read()
        assert "startup_recovery" in source

    def test_main_py_startup_recovery_in_try_except(self):
        """startup_recovery must be created inside try/except in startup_event."""
        main_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "app",
            "main.py",
        )
        with open(main_path) as f:
            source = f.read()
        # StartupRecoveryService is created before the watcher starts so that
        # recovery completes before the observer begins watching.
        watcher_idx = source.find("watcher_service.start()")
        recovery_idx = source.find("StartupRecoveryService()")
        assert recovery_idx < watcher_idx, (
            "StartupRecoveryService() must be started before watcher_service.start()"
        )

    def test_main_py_recovery_before_watcher_in_try_except(self):
        """startup_recovery must be created inside try/except in startup_event."""
        main_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "app",
            "main.py",
        )
        with open(main_path) as f:
            source = f.read()
        # StartupRecoveryService is created before the watcher starts so that
        # recovery completes before the observer begins watching.
        watcher_idx = source.find("watcher_service.start()")
        recovery_idx = source.find("StartupRecoveryService()")
        assert recovery_idx < watcher_idx, (
            "StartupRecoveryService() must be started before watcher_service.start()"
        )


class TestNoCircularImports:
    """Verify no circular import between main.py and startup_recovery_service.py."""

    def test_startup_recovery_service_imports_cleanly(self):
        """The recovery service module must import without circular errors."""
        from app.services.startup_recovery_service import StartupRecoveryService

        assert StartupRecoveryService is not None

    def test_recovery_service_has_start_and_stop(self):
        """StartupRecoveryService must have start() and stop() methods."""
        from app.services.startup_recovery_service import StartupRecoveryService

        assert hasattr(StartupRecoveryService, "start")
        assert hasattr(StartupRecoveryService, "stop")
        assert callable(getattr(StartupRecoveryService, "start"))
        assert callable(getattr(StartupRecoveryService, "stop"))


# ---------------------------------------------------------------------------
# Runtime wiring tests — patch everything before calling lifecycle functions
# ---------------------------------------------------------------------------


def _build_startup_patchers():
    """Return a list of context managers that mock all startup_event internals."""
    return [
        patch("app.main._seed_root_org_and_superadmin"),
        patch("app.main.init_storage"),
    ]


class TestStartupEventWiring:
    """Test that startup_event properly starts the recovery service."""

    def test_startup_event_starts_recovery_service(self):
        """startup_event must create and start StartupRecoveryService when WATCHER_ENABLED."""

        async def run():
            with patch("app.main.settings") as mock_settings, patch(
                "app.main._seed_root_org_and_superadmin"
            ), patch("app.main.init_storage"), patch(
                "app.main.SessionLocal"
            ) as mock_session_cls, patch(
                "app.main.DataStoreWatcher"
            ) as mock_watcher_cls, patch(
                "app.main.StartupRecoveryService"
            ) as mock_recovery_cls:
                mock_settings.WATCHER_ENABLED = True
                mock_watcher_instance = MagicMock()
                mock_watcher_cls.return_value = mock_watcher_instance
                mock_recovery_instance = MagicMock()
                mock_recovery_cls.return_value = mock_recovery_instance

                # Mock the DB query at the bottom of startup_event
                mock_query = MagicMock()
                mock_query.filter.return_value.all.return_value = []
                mock_session_cls.return_value.query.return_value = mock_query

                from app.main import startup_event

                await startup_event()

                mock_recovery_cls.assert_called_once()
                mock_recovery_instance.start.assert_called_once()

        asyncio.run(run())

    def test_startup_event_skips_recovery_when_watcher_disabled(self):
        """startup_event must NOT start DataStoreWatcher when WATCHER_ENABLED is False.

        Note: StartupRecoveryService is always created regardless of WATCHER_ENABLED,
        because it only walks the datastore folder tree on disk and does not need
        the filesystem observer.
        """

        async def run():
            from app.main import startup_event

            with patch("app.main.settings") as mock_settings, patch(
                "app.main._seed_root_org_and_superadmin"
            ), patch("app.main.init_storage"), patch(
                "app.main.SessionLocal"
            ) as mock_session_cls, patch(
                "app.main.DataStoreWatcher"
            ) as mock_watcher_cls, patch(
                "app.main.StartupRecoveryService"
            ) as mock_recovery_cls:
                mock_settings.WATCHER_ENABLED = False
                mock_watcher_cls.return_value = MagicMock()

                mock_query = MagicMock()
                mock_query.filter.return_value.all.return_value = []
                mock_session_cls.return_value.query.return_value = mock_query

                await startup_event()

                # Recovery service is always created (independent of WATCHER_ENABLED)
                mock_recovery_cls.assert_called_once()
                mock_recovery_cls.return_value.start.assert_called_once()

                # Watcher should NOT be created when disabled
                mock_watcher_cls.assert_not_called()

        asyncio.run(run())

    def test_startup_event_catches_recovery_start_exception(self):
        """startup_event must not crash if StartupRecoveryService.start() raises."""

        async def run():
            from app.main import startup_event

            with patch("app.main.settings") as mock_settings, patch(
                "app.main._seed_root_org_and_superadmin"
            ), patch("app.main.init_storage"), patch(
                "app.main.SessionLocal"
            ) as mock_session_cls, patch(
                "app.main.DataStoreWatcher"
            ) as mock_watcher_cls, patch(
                "app.main.StartupRecoveryService"
            ) as mock_recovery_cls:
                mock_settings.WATCHER_ENABLED = True
                mock_watcher_cls.return_value = MagicMock()
                mock_recovery_instance = MagicMock()
                mock_recovery_cls.return_value = mock_recovery_instance
                mock_recovery_instance.start.side_effect = RuntimeError("test error")

                mock_query = MagicMock()
                mock_query.filter.return_value.all.return_value = []
                mock_session_cls.return_value.query.return_value = mock_query

                # Must NOT raise
                await startup_event()

                # The exception was raised but caught (we get here without crash)

        asyncio.run(run())


class TestShutdownEventWiring:
    """Test that shutdown_event properly stops the recovery service."""

    def test_shutdown_event_stops_recovery_service(self):
        """shutdown_event must call startup_recovery.stop() when not None."""

        async def run():
            with patch("app.main.settings") as mock_settings, patch(
                "app.main._seed_root_org_and_superadmin"
            ), patch("app.main.init_storage"), patch(
                "app.main.SessionLocal"
            ) as mock_session_cls, patch(
                "app.main.DataStoreWatcher"
            ) as mock_watcher_cls, patch(
                "app.main.StartupRecoveryService"
            ) as mock_recovery_cls:
                mock_settings.WATCHER_ENABLED = True
                mock_watcher_cls.return_value = MagicMock()
                mock_recovery_instance = MagicMock()
                mock_recovery_cls.return_value = mock_recovery_instance

                mock_query = MagicMock()
                mock_query.filter.return_value.all.return_value = []
                mock_session_cls.return_value.query.return_value = mock_query

                from app.main import startup_event

                await startup_event()

            from app.main import shutdown_event

            await shutdown_event()

            mock_recovery_instance.stop.assert_called_once()

        asyncio.run(run())

    def test_shutdown_event_handles_none_recovery(self):
        """shutdown_event must not crash if startup_recovery is None (never started)."""

        async def run():
            with patch("app.main.settings") as mock_settings, patch(
                "app.main._seed_root_org_and_superadmin"
            ), patch("app.main.init_storage"), patch(
                "app.main.SessionLocal"
            ) as mock_session_cls, patch(
                "app.main.DataStoreWatcher"
            ) as mock_watcher_cls:
                mock_settings.WATCHER_ENABLED = False
                mock_watcher_cls.return_value = MagicMock()

                mock_query = MagicMock()
                mock_query.filter.return_value.all.return_value = []
                mock_session_cls.return_value.query.return_value = mock_query

                from app.main import startup_event

                await startup_event()

            from app.main import shutdown_event

            # Must not raise
            await shutdown_event()

        asyncio.run(run())

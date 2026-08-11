"""Tests for wiring StartupRecoveryService into main.py lifespan.

Verifies:
1. main.py imports cleanly (syntax + import validation).
2. _services module-level dict exists in main.py.
3. StartupRecoveryService start/stop are wired in lifespan.
4. lifespan respects WATCHER_ENABLED gating.
5. lifespan handles _services being empty (null safety).
6. No circular import errors.
"""
import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

# conftest.py sets up the SQLite session before any app.* import.


@pytest.fixture(autouse=True)
def reset_main_globals():
    """Reset module-level globals in main.py before each test.

    lifespan mutates the module-level `_services` dict. Without
    resetting between tests, later tests see stale values from earlier tests.
    """
    import app.main

    app.main._services["watcher"] = None
    app.main._services["recovery"] = None
    yield
    app.main._services["watcher"] = None
    app.main._services["recovery"] = None


# ---------------------------------------------------------------------------
# Static structure tests — no runtime calls into main.py functions
# ---------------------------------------------------------------------------


class TestMainPyImportAndSyntax:
    """Test that main.py has valid Python syntax and the right structure."""

    def test_main_py_syntax_is_valid(self):
        """main.py must parse without SyntaxError."""
        import ast
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
            "from app.services.discovery import StartupRecoveryService"
            in source
        )

    def test_main_py_has_services_dict(self):
        """main.py must declare _services dict."""
        main_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "app",
            "main.py",
        )
        with open(main_path) as f:
            source = f.read()
        assert "_services" in source

    def test_main_py_shutdown_handles_recovery(self):
        """main.py lifespan must reference _services['recovery'].stop()."""
        main_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "app",
            "main.py",
        )
        with open(main_path) as f:
            source = f.read()
        assert "_services['recovery'].stop()" in source or "_services[\"recovery\"].stop()" in source

    def test_main_py_recovery_before_watcher_in_try_except(self):
        """startup recovery must be created before watcher in lifespan."""
        main_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "app",
            "main.py",
        )
        with open(main_path) as f:
            source = f.read()
        watcher_idx = source.find("watcher_service.start()")
        if watcher_idx == -1:
            watcher_idx = source.find("_services['watcher'].start()")
        if watcher_idx == -1:
            watcher_idx = source.find('_services["watcher"].start()')
        recovery_idx = source.find("StartupRecoveryService()")
        assert recovery_idx > 0 and watcher_idx > 0, (
            "Both StartupRecoveryService() and watcher start must be present"
        )
        assert recovery_idx < watcher_idx, (
            "StartupRecoveryService() must be started before watcher_service.start()"
        )


class TestNoCircularImports:
    """Verify no circular import between main.py and startup_recovery_service.py."""

    def test_startup_recovery_service_imports_cleanly(self):
        """The recovery service module must import without circular errors."""
        from app.services.discovery import StartupRecoveryService

        assert StartupRecoveryService is not None

    def test_recovery_service_has_start_and_stop(self):
        """StartupRecoveryService must have start() and stop() methods."""
        from app.services.discovery import StartupRecoveryService

        assert hasattr(StartupRecoveryService, "start")
        assert hasattr(StartupRecoveryService, "stop")
        assert callable(getattr(StartupRecoveryService, "start"))
        assert callable(getattr(StartupRecoveryService, "stop"))


# ---------------------------------------------------------------------------
# Runtime wiring tests — patch everything before calling lifespan
# ---------------------------------------------------------------------------


class TestLifespanWiring:
    """Test that lifespan properly starts and stops the recovery service."""

    def test_lifespan_starts_recovery_service(self):
        """lifespan must create and start StartupRecoveryService when WATCHER_ENABLED."""

        async def run():
            with patch("app.main.settings") as mock_settings, patch(
                "app.main._seed_root_org_and_superadmin"
            ), patch("app.main.init_storage"), patch(
                "app.main.SessionLocal"
            ) as mock_session_cls, patch(
                "app.main.DataStoreWatcher"
            ) as mock_watcher_cls, patch(
                "app.main.StartupRecoveryService"
            ) as mock_recovery_cls, patch(
                "app.services.settings_service.get_setting"
            ) as mock_get_setting:
                mock_get_setting.return_value = True
                mock_watcher_instance = MagicMock()
                mock_watcher_cls.return_value = mock_watcher_instance
                mock_recovery_instance = MagicMock()
                mock_recovery_cls.return_value = mock_recovery_instance

                # Mock the DB query at the bottom of lifespan
                mock_query = MagicMock()
                mock_query.filter.return_value.all.return_value = []
                mock_session_cls.return_value.query.return_value = mock_query

                import app.main as main_module
                from app.main import app

                # Call the lifespan generator directly
                gen = main_module.lifespan(app)
                try:
                    await gen.__anext__()
                    mock_recovery_cls.assert_called_once()
                    mock_recovery_instance.start.assert_called_once()
                finally:
                    try:
                        await gen.__anext__()
                    except StopAsyncIteration:
                        pass

        asyncio.run(run())

    def test_lifespan_skips_watcher_when_disabled(self):
        """lifespan must NOT start DataStoreWatcher when WATCHER_ENABLED is False."""

        async def run():
            with patch("app.main.settings") as mock_settings, patch(
                "app.main._seed_root_org_and_superadmin"
            ), patch("app.main.init_storage"), patch(
                "app.main.SessionLocal"
            ) as mock_session_cls, patch(
                "app.main.DataStoreWatcher"
            ) as mock_watcher_cls, patch(
                "app.main.StartupRecoveryService"
            ) as mock_recovery_cls, patch(
                "app.services.settings_service.get_setting"
            ) as mock_get_setting:
                mock_get_setting.return_value = False
                mock_watcher_cls.return_value = MagicMock()

                mock_query = MagicMock()
                mock_query.filter.return_value.all.return_value = []
                mock_session_cls.return_value.query.return_value = mock_query

                import app.main as main_module
                from app.main import app

                gen = main_module.lifespan(app)
                try:
                    await gen.__anext__()
                    # Recovery service is always created (independent of WATCHER_ENABLED)
                    mock_recovery_cls.assert_called_once()
                    mock_recovery_cls.return_value.start.assert_called_once()

                    # Watcher should NOT be created when disabled
                    mock_watcher_cls.assert_not_called()
                finally:
                    try:
                        await gen.__anext__()
                    except StopAsyncIteration:
                        pass

        asyncio.run(run())

    def test_lifespan_catches_recovery_start_exception(self):
        """lifespan must not crash if StartupRecoveryService.start() raises."""

        async def run():
            with patch("app.main.settings") as mock_settings, patch(
                "app.main._seed_root_org_and_superadmin"
            ), patch("app.main.init_storage"), patch(
                "app.main.SessionLocal"
            ) as mock_session_cls, patch(
                "app.main.DataStoreWatcher"
            ) as mock_watcher_cls, patch(
                "app.main.StartupRecoveryService"
            ) as mock_recovery_cls, patch(
                "app.services.settings_service.get_setting"
            ) as mock_get_setting:
                mock_get_setting.return_value = True
                mock_watcher_cls.return_value = MagicMock()
                mock_recovery_instance = MagicMock()
                mock_recovery_cls.return_value = mock_recovery_instance
                mock_recovery_instance.start.side_effect = RuntimeError("test error")

                mock_query = MagicMock()
                mock_query.filter.return_value.all.return_value = []
                mock_session_cls.return_value.query.return_value = mock_query

                import app.main as main_module
                from app.main import app

                gen = main_module.lifespan(app)
                try:
                    # Must NOT raise
                    await gen.__anext__()
                finally:
                    try:
                        await gen.__anext__()
                    except StopAsyncIteration:
                        pass

        asyncio.run(run())

    def test_lifespan_stops_recovery_service(self):
        """lifespan must call _services['recovery'].stop() on shutdown."""

        async def run():
            with patch("app.main.settings") as mock_settings, patch(
                "app.main._seed_root_org_and_superadmin"
            ), patch("app.main.init_storage"), patch(
                "app.main.SessionLocal"
            ) as mock_session_cls, patch(
                "app.main.DataStoreWatcher"
            ) as mock_watcher_cls, patch(
                "app.main.StartupRecoveryService"
            ) as mock_recovery_cls, patch(
                "app.services.settings_service.get_setting"
            ) as mock_get_setting:
                mock_get_setting.return_value = True
                mock_watcher_cls.return_value = MagicMock()
                mock_recovery_instance = MagicMock()
                mock_recovery_cls.return_value = mock_recovery_instance

                mock_query = MagicMock()
                mock_query.filter.return_value.all.return_value = []
                mock_session_cls.return_value.query.return_value = mock_query

                import app.main as main_module
                from app.main import app

                gen = main_module.lifespan(app)
                try:
                    await gen.__anext__()
                finally:
                    try:
                        await gen.__anext__()
                    except StopAsyncIteration:
                        pass

                mock_recovery_instance.stop.assert_called_once()

        asyncio.run(run())

    def test_lifespan_handles_none_recovery(self):
        """lifespan must not crash if _services['recovery'] is None (never started)."""

        async def run():
            with patch("app.main.settings") as mock_settings, patch(
                "app.main._seed_root_org_and_superadmin"
            ), patch("app.main.init_storage"), patch(
                "app.main.SessionLocal"
            ) as mock_session_cls, patch(
                "app.main.DataStoreWatcher"
            ) as mock_watcher_cls, patch(
                "app.services.settings_service.get_setting"
            ) as mock_get_setting:
                mock_get_setting.return_value = False
                mock_watcher_cls.return_value = MagicMock()

                mock_query = MagicMock()
                mock_query.filter.return_value.all.return_value = []
                mock_session_cls.return_value.query.return_value = mock_query

                import app.main as main_module
                from app.main import app

                gen = main_module.lifespan(app)
                try:
                    # Must not raise
                    await gen.__anext__()
                finally:
                    try:
                        await gen.__anext__()
                    except StopAsyncIteration:
                        pass

        asyncio.run(run())

from __future__ import annotations

import signal

from arc_llm import install_signal_cancel_chain


def test_signal_cancel_chain_combines_parent_check_and_restores_handlers():
    previous = signal.getsignal(signal.SIGTERM)
    parent = {"cancelled": False}

    with install_signal_cancel_chain(lambda: parent["cancelled"]) as cancel_check:
        assert cancel_check() is False
        parent["cancelled"] = True
        assert cancel_check() is True
        parent["cancelled"] = False

        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(signal.SIGTERM, None)
        assert cancel_check() is True

    assert signal.getsignal(signal.SIGTERM) == previous

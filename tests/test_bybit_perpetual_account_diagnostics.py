from __future__ import annotations

import pytest

from tradingagents.crypto.bybit_perpetual_account import (
    BybitLinearDemoPositionClient,
    BybitPerpetualAccountError,
)


class Response:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


def test_bybit_ret_message_is_exposed_without_credentials():
    client = BybitLinearDemoPositionClient(
        api_key="demo-key-secret",
        api_secret="demo-api-secret",
        transport=lambda **kwargs: Response(
            {"retCode": 10003, "retMsg": "API key is invalid demo-key-secret"}
        ),
        clock_ms=lambda: 1_700_000_000_000,
    )

    with pytest.raises(BybitPerpetualAccountError) as error:
        client.get_wallet_balance()

    message = str(error.value)
    assert "retCode 10003" in message
    assert "API key is invalid" in message
    assert "demo-key-secret" not in message
    assert "demo-api-secret" not in message


def test_malformed_exchange_message_is_bounded_and_single_line():
    long_message = "exchange rejected " + ("x" * 500)
    client = BybitLinearDemoPositionClient(
        api_key="key",
        api_secret="secret",
        transport=lambda **kwargs: Response(
            {"retCode": 10004, "retMsg": long_message + "\\nforged-log-line"}
        ),
        clock_ms=lambda: 1_700_000_000_000,
    )

    with pytest.raises(BybitPerpetualAccountError) as error:
        client.get_wallet_balance()

    message = str(error.value)
    assert "retCode 10004" in message
    assert len(message) < 230
    assert "\\n" not in message

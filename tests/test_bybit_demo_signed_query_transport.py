import requests

from tradingagents.paper_experiment.broker.bybit_demo import _default_transport


def test_default_transport_does_not_append_signed_query_parameters_twice(monkeypatch):
    captured = {}

    class Response:
        status_code = 200
        text = '{"retCode":0,"retMsg":"OK"}'

    def fake_request(method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["kwargs"] = kwargs
        return Response()

    monkeypatch.setattr(requests, "request", fake_request)

    response = _default_transport(
        method="GET",
        url="https://api-demo.bybit.com/v5/account/wallet-balance",
        headers={"X-BAPI-SIGN": "test-signature"},
        params={"accountType": "UNIFIED"},
        json_body=None,
        timeout=5,
    )

    assert response.status_code == 200
    assert captured["url"].endswith("?accountType=UNIFIED")
    assert captured["url"].count("accountType=") == 1
    assert "params" not in captured["kwargs"]


def test_default_transport_preserves_encoded_signed_query_string(monkeypatch):
    captured = {}

    class Response:
        status_code = 200
        text = '{"retCode":0,"retMsg":"OK"}'

    def fake_request(method, url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return Response()

    monkeypatch.setattr(requests, "request", fake_request)

    _default_transport(
        method="GET",
        url="https://api-demo.bybit.com/v5/order/realtime",
        headers={},
        params={"category": "spot", "openOnly": 1},
        json_body=None,
        timeout=5,
    )

    assert captured["url"].endswith("?category=spot&openOnly=1")
    assert "params" not in captured["kwargs"]

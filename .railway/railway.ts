import { defineRailway, project, service, volume } from "railway/iac";

/**
 * The Crypto TJR dashboard, as Railway Infrastructure as Code.
 *
 * This file replaces `railway.json`: Railway deprecated Config as Code and stops
 * reading it on 2026-12-01, so the JSON file is a short-lived bridge and this
 * is the file that will still be read then.
 *
 * The two files describe the same service, and a test pins that they agree —
 * a start command or a health check that differs between them is a deployment
 * that behaves differently from what the repository says it does.
 *
 * `--allow-paper-trading` only *permits* the dashboard's START button to build
 * a paper worker; the worker starts only when an authenticated operator presses
 * it, and the broker it may reach is Alpaca's **paper** endpoint (the live host
 * is refused in code). Once the operator presses START, that worker does place
 * paper orders — this configuration enables that path rather than merely
 * describing a simulation.
 */
export default defineRailway(() => {
  // The worker's durable state — the processed/ordered decision sets, the
  // experiment journal and the source state files — lives under the experiment
  // root. Without a volume that directory is ephemeral, and a redeploy would
  // forget which decisions already produced an order, so the same decision
  // could be placed twice. The volume makes a restart idempotent.
  const data = volume("data", { sizeMB: 1024 });

  const dashboard = service("crypto-dashboard", {
    // The service is built from the repository's Dockerfile, the same builder
    // `railway.json` names, so the two files describe one deployment.
    build: { builder: "DOCKERFILE", dockerfilePath: "Dockerfile" },
    // The start command is wrapped in a shell because Railway runs a Dockerfile
    // service's start command in exec form, where `$PORT` is not expanded.
    start:
      '/bin/sh -c "exec python -m tradingagents.paper_experiment.dashboard.server' +
      " --root /home/appuser/app/data --host 0.0.0.0 --port $PORT" +
      " --allow-paper-trading --paper-market-data binance_snapshot" +
      ' --paper-decision-source crypto --crypto-live"',
    healthcheck: "/healthz",
    healthcheckTimeout: 30,
    volumeMounts: {
      "/home/appuser/app/data": data,
    },
    env: {
      PYTHONUNBUFFERED: "1",
      PYTHONDONTWRITEBYTECODE: "1",
      // The image runs as the non-root `appuser`, which cannot write to a
      // Railway volume mount: the first write to the volume fails with
      // PermissionError. Railway's own remedy is to run the container as root,
      // which is what this variable selects.
      RAILWAY_RUN_UID: "0",
      // The dashboard token, the Alpaca paper credentials and any LLM key are
      // set in the Railway dashboard, not here: a secret in this file would be
      // a secret in the repository.
      TRADINGAGENTS_CRYPTO_SYMBOLS:
        "BTC/USDT ETH/USDT SOL/USDT XRP/USDT BNB/USDT DOGE/USDT AVAX/USDT LINK/USDT LTC/USDT BCH/USDT",
      TRADINGAGENTS_EXPERIMENT_SYMBOLS:
        "BTC/USDT ETH/USDT SOL/USDT XRP/USDT BNB/USDT DOGE/USDT AVAX/USDT LINK/USDT LTC/USDT BCH/USDT",
      TRADINGAGENTS_PAPER_MARKET_DATA: "binance_snapshot",
      TRADINGAGENTS_PAPER_DECISION_SOURCE: "crypto",
      TRADINGAGENTS_PAPER_ORDER_NOTIONAL: "25",
      TRADINGAGENTS_PAPER_MAX_PRICE_DEVIATION: "0.05",
      TRADINGAGENTS_PAPER_SNAPSHOT_MAX_AGE_SECONDS: "15",
    },
  });

  return project("tradingagents", { resources: [dashboard, data] });
});

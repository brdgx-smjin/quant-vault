module.exports = {
  apps: [
    {
      name: "quant-discord-bot",
      script: ".venv/bin/python",
      args: "scripts/discord_bot.py",
      cwd: "/Users/jimmy/quant/quant-vault",
      autorestart: true,
      max_restarts: 10,
      restart_delay: 5000,
      env: {
        PYTHONUNBUFFERED: "1",
        SSL_CERT_FILE:
          "/Users/jimmy/quant/quant-vault/.venv/lib/python3.12/site-packages/certifi/cacert.pem",
      },
    },
  ],
};

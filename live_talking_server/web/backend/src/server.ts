import { buildApp } from "./app";

async function startServer(): Promise<void> {
  const app = buildApp();
  const port = Number(process.env.PORT ?? 3000);

  await app.listen({
    host: "0.0.0.0",
    port
  });
}

void startServer();
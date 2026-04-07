import Fastify, { type FastifyInstance } from "fastify";
import cors from "@fastify/cors";

import { assistantRoutes } from "./modules/assistant/assistant.routes";
import { taskRoutes } from "./modules/tasks/task.routes";
import { sseRoutes } from "./modules/realtime/sse.routes";
import { AppError } from "./shared/errors/app-error";
import { type ApiResponse } from "./shared/types/common";

export function buildApp(): FastifyInstance {
  const app = Fastify({
    logger: true
  });

  app.register(cors, {
    origin: true
  });

  app.setErrorHandler((error, request, reply) => {
    const appError = error instanceof AppError ? error : null;
    const statusCode = appError?.statusCode ?? 500;
    const response: ApiResponse<never> = {
      success: false,
      requestId: request.id,
      error: {
        code: appError?.code ?? "INTERNAL_ERROR",
        message: error instanceof Error ? error.message : "Unexpected error.",
        details: appError?.details
      }
    };

    void reply.code(statusCode).send(response);
  });

  app.register(assistantRoutes, { prefix: "/api/assistant" });
  app.register(taskRoutes, { prefix: "/api/tasks" });
  app.register(sseRoutes, { prefix: "/api/tasks" });

  return app;
}

import { type FastifyPluginAsync } from "fastify";

import { realtimeService } from "./realtime.service";

export const sseRoutes: FastifyPluginAsync = async (app) => {
  app.get<{ Params: { taskId: string } }>(
    "/:taskId/stream",
    async (request, reply) => {
      reply.hijack();

      const response = reply.raw;
      const { taskId } = request.params;

      response.setHeader("Content-Type", "text/event-stream; charset=utf-8");
      response.setHeader("Cache-Control", "no-cache, no-transform");
      response.setHeader("Connection", "keep-alive");
      response.setHeader("X-Accel-Buffering", "no");
      response.setHeader("Access-Control-Allow-Origin", "*");
      response.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
      response.setHeader("Access-Control-Allow-Headers", "Content-Type");
      response.flushHeaders?.();
      response.write("retry: 3000\n");
      response.write(": connected\n\n");

      await realtimeService.subscribeSSE(taskId, response);

      const heartbeat = setInterval(() => {
        if (!response.writableEnded) {
          response.write(": keep-alive\n\n");
        }
      }, 15000);

      const cleanup = (): void => {
        clearInterval(heartbeat);
        realtimeService.unsubscribe(taskId, response);
      };

      request.raw.on("close", cleanup);
      response.on("close", cleanup);
      response.on("error", cleanup);
    }
  );
};

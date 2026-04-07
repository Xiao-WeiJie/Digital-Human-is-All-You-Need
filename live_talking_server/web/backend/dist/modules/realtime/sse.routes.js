"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.sseRoutes = void 0;
const realtime_service_1 = require("./realtime.service");
const sseRoutes = async (app) => {
    app.get("/:taskId/stream", async (request, reply) => {
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
        await realtime_service_1.realtimeService.subscribeSSE(taskId, response);
        const heartbeat = setInterval(() => {
            if (!response.writableEnded) {
                response.write(": keep-alive\n\n");
            }
        }, 15000);
        const cleanup = () => {
            clearInterval(heartbeat);
            realtime_service_1.realtimeService.unsubscribe(taskId, response);
        };
        request.raw.on("close", cleanup);
        response.on("close", cleanup);
        response.on("error", cleanup);
    });
};
exports.sseRoutes = sseRoutes;

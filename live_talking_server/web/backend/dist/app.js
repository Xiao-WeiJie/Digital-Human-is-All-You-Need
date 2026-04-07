"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.buildApp = buildApp;
const fastify_1 = __importDefault(require("fastify"));
const cors_1 = __importDefault(require("@fastify/cors"));
const assistant_routes_1 = require("./modules/assistant/assistant.routes");
const task_routes_1 = require("./modules/tasks/task.routes");
const sse_routes_1 = require("./modules/realtime/sse.routes");
const app_error_1 = require("./shared/errors/app-error");
function buildApp() {
    const app = (0, fastify_1.default)({
        logger: true
    });
    app.register(cors_1.default, {
        origin: true
    });
    app.setErrorHandler((error, request, reply) => {
        const appError = error instanceof app_error_1.AppError ? error : null;
        const statusCode = appError?.statusCode ?? 500;
        const response = {
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
    app.register(assistant_routes_1.assistantRoutes, { prefix: "/api/assistant" });
    app.register(task_routes_1.taskRoutes, { prefix: "/api/tasks" });
    app.register(sse_routes_1.sseRoutes, { prefix: "/api/tasks" });
    return app;
}

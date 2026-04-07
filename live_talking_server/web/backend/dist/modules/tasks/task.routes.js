"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.taskRoutes = void 0;
const task_service_1 = require("./task.service");
function resolveUserId(rawUserId) {
    if (Array.isArray(rawUserId)) {
        return rawUserId[0] ?? "demo-user";
    }
    return rawUserId ?? "demo-user";
}
const taskRoutes = async (app) => {
    app.get("/:taskId/events", async (request) => {
        const userId = resolveUserId(request.headers["x-user-id"]);
        const rawAfterSequence = Number(request.query.afterSequence ?? 0);
        const afterSequence = Number.isFinite(rawAfterSequence) && rawAfterSequence > 0 ? Math.floor(rawAfterSequence) : 0;
        const eventResponse = await task_service_1.taskService.getTaskEventsResponse(request.params.taskId, userId, afterSequence);
        const response = {
            success: true,
            requestId: request.id,
            data: eventResponse
        };
        return response;
    });
    app.get("/:taskId", async (request) => {
        const userId = resolveUserId(request.headers["x-user-id"]);
        const task = await task_service_1.taskService.getTask(request.params.taskId, userId);
        const recentEvents = await task_service_1.taskService.getTaskEvents(request.params.taskId, userId);
        const candidates = await task_service_1.taskService.getTaskCandidates(request.params.taskId, userId);
        const pendingHumanAction = await task_service_1.taskService.getPendingHumanAction(request.params.taskId, userId);
        const response = {
            success: true,
            requestId: request.id,
            data: {
                task,
                checkpoints: [],
                recentEvents,
                candidates,
                pendingHumanAction
            }
        };
        return response;
    });
    app.post("/:taskId/confirm-intent", async (request) => {
        const userId = resolveUserId(request.headers["x-user-id"]);
        const task = await task_service_1.taskService.confirmIntent(request.params.taskId, request.body, userId);
        const response = {
            success: true,
            requestId: request.id,
            data: {
                taskId: task.id,
                status: task.status
            }
        };
        return response;
    });
    app.post("/:taskId/confirm-candidate", async (request) => {
        const userId = resolveUserId(request.headers["x-user-id"]);
        const task = await task_service_1.taskService.confirmCandidate(request.params.taskId, request.body, userId);
        const response = {
            success: true,
            requestId: request.id,
            data: {
                taskId: task.id,
                status: task.status
            }
        };
        return response;
    });
    app.post("/:taskId/human-step-done", async (request) => {
        const userId = resolveUserId(request.headers["x-user-id"]);
        const task = await task_service_1.taskService.markHumanStepDone(request.params.taskId, request.body, userId);
        const response = {
            success: true,
            requestId: request.id,
            data: {
                taskId: task.id,
                status: task.status
            }
        };
        return response;
    });
    app.post("/:taskId/confirm-submit", async (request) => {
        const userId = resolveUserId(request.headers["x-user-id"]);
        const task = await task_service_1.taskService.confirmSubmit(request.params.taskId, request.body, userId);
        const response = {
            success: true,
            requestId: request.id,
            data: {
                taskId: task.id,
                status: task.status
            }
        };
        return response;
    });
    app.post("/:taskId/complete-payment", async (request) => {
        const userId = resolveUserId(request.headers["x-user-id"]);
        const task = await task_service_1.taskService.completePayment(request.params.taskId, request.body, userId);
        const response = {
            success: true,
            requestId: request.id,
            data: {
                taskId: task.id,
                status: task.status
            }
        };
        return response;
    });
    app.post("/:taskId/cancel", async (request) => {
        const userId = resolveUserId(request.headers["x-user-id"]);
        const task = await task_service_1.taskService.cancelTask(request.params.taskId, request.body, userId);
        const response = {
            success: true,
            requestId: request.id,
            data: {
                taskId: task.id,
                status: task.status
            }
        };
        return response;
    });
};
exports.taskRoutes = taskRoutes;

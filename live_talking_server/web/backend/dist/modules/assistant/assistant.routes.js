"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.assistantRoutes = void 0;
const task_enums_1 = require("../tasks/task.enums");
const task_service_1 = require("../tasks/task.service");
const intent_parser_service_1 = require("../booking/intent-parser.service");
function resolveUserId(rawUserId) {
    if (Array.isArray(rawUserId)) {
        return rawUserId[0] ?? "demo-user";
    }
    return rawUserId ?? "demo-user";
}
const assistantRoutes = async (app) => {
    app.post("/tasks", async (request, reply) => {
        const userId = resolveUserId(request.headers["x-user-id"]);
        const task = await task_service_1.taskService.createTask(request.body, userId);
        await task_service_1.taskService.transition(task.id, task_enums_1.AgentTaskStatus.PARSING_INTENT);
        const mockParseResult = (0, intent_parser_service_1.mockParseIntent)(request.body.rawText, task.id);
        const nextStatus = mockParseResult.needsClarification
            ? task_enums_1.AgentTaskStatus.AWAITING_SLOT_CLARIFICATION
            : task_enums_1.AgentTaskStatus.AWAITING_INTENT_CONFIRMATION;
        await task_service_1.taskService.transition(task.id, nextStatus, {
            intent: mockParseResult.intent,
            context: mockParseResult.context,
            eventMessage: mockParseResult.needsClarification
                ? "Mock 解析完成，等待用户补充缺失槽位"
                : "Mock 解析完成，等待用户确认订票意图",
            eventPayload: {
                missingSlots: mockParseResult.context.missingSlots,
                ambiguousSlots: mockParseResult.context.ambiguousSlots
            }
        });
        const response = {
            success: true,
            requestId: request.id,
            data: {
                taskId: task.id,
                status: nextStatus,
                intent: mockParseResult.intent,
                missingSlots: mockParseResult.context.missingSlots,
                ambiguousSlots: mockParseResult.context.ambiguousSlots
            }
        };
        return reply.code(201).send(response);
    });
};
exports.assistantRoutes = assistantRoutes;

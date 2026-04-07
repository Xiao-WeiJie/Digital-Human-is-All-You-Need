import { type FastifyPluginAsync } from "fastify";

import { AgentTaskStatus } from "../tasks/task.enums";
import { taskService } from "../tasks/task.service";
import { type CreateAssistantTaskRequest } from "../tasks/task.types";
import { mockParseIntent } from "../booking/intent-parser.service";
import { type ApiResponse } from "../../shared/types/common";

type CreateAssistantTaskResponse = {
  taskId: string;
  status: AgentTaskStatus;
  intent: ReturnType<typeof mockParseIntent>["intent"];
  missingSlots: ReturnType<typeof mockParseIntent>["context"]["missingSlots"];
  ambiguousSlots: ReturnType<typeof mockParseIntent>["context"]["ambiguousSlots"];
};

function resolveUserId(rawUserId: string | string[] | undefined): string {
  if (Array.isArray(rawUserId)) {
    return rawUserId[0] ?? "demo-user";
  }

  return rawUserId ?? "demo-user";
}

export const assistantRoutes: FastifyPluginAsync = async (app) => {
  app.post<{ Body: CreateAssistantTaskRequest }>(
    "/tasks",
    async (request, reply) => {
      const userId = resolveUserId(request.headers["x-user-id"]);
      const task = await taskService.createTask(request.body, userId);

      await taskService.transition(task.id, AgentTaskStatus.PARSING_INTENT);

      const mockParseResult = mockParseIntent(request.body.rawText, task.id);
      const nextStatus = mockParseResult.needsClarification
        ? AgentTaskStatus.AWAITING_SLOT_CLARIFICATION
        : AgentTaskStatus.AWAITING_INTENT_CONFIRMATION;

      await taskService.transition(task.id, nextStatus, {
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

      const response: ApiResponse<CreateAssistantTaskResponse> = {
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
    }
  );
};

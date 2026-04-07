import { type FastifyPluginAsync } from "fastify";

import { type ApiResponse } from "../../shared/types/common";
import { taskService } from "./task.service";
import {
  type CancelTaskRequest,
  type CompletePaymentRequest,
  type ConfirmCandidateRequest,
  type ConfirmIntentRequest,
  type ConfirmSubmitRequest,
  type GetTaskEventsResponse,
  type GetTaskDetailResponse,
  type HumanStepDoneRequest
} from "./task.types";

type TaskActionResponse = {
  taskId: string;
  status: string;
};

function resolveUserId(rawUserId: string | string[] | undefined): string {
  if (Array.isArray(rawUserId)) {
    return rawUserId[0] ?? "demo-user";
  }

  return rawUserId ?? "demo-user";
}

export const taskRoutes: FastifyPluginAsync = async (app) => {
  app.get<{ Params: { taskId: string }; Querystring: { afterSequence?: string } }>(
    "/:taskId/events",
    async (request) => {
      const userId = resolveUserId(request.headers["x-user-id"]);
      const rawAfterSequence = Number(request.query.afterSequence ?? 0);
      const afterSequence =
        Number.isFinite(rawAfterSequence) && rawAfterSequence > 0 ? Math.floor(rawAfterSequence) : 0;
      const eventResponse = await taskService.getTaskEventsResponse(
        request.params.taskId,
        userId,
        afterSequence
      );

      const response: ApiResponse<GetTaskEventsResponse> = {
        success: true,
        requestId: request.id,
        data: eventResponse
      };

      return response;
    }
  );

  app.get<{ Params: { taskId: string } }>(
    "/:taskId",
    async (request) => {
      const userId = resolveUserId(request.headers["x-user-id"]);
      const task = await taskService.getTask(request.params.taskId, userId);
      const recentEvents = await taskService.getTaskEvents(request.params.taskId, userId);
      const candidates = await taskService.getTaskCandidates(request.params.taskId, userId);
      const pendingHumanAction = await taskService.getPendingHumanAction(request.params.taskId, userId);

      const response: ApiResponse<GetTaskDetailResponse> = {
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
    }
  );

  app.post<{ Params: { taskId: string }; Body: ConfirmIntentRequest }>(
    "/:taskId/confirm-intent",
    async (request) => {
      const userId = resolveUserId(request.headers["x-user-id"]);
      const task = await taskService.confirmIntent(request.params.taskId, request.body, userId);

      const response: ApiResponse<TaskActionResponse> = {
        success: true,
        requestId: request.id,
        data: {
          taskId: task.id,
          status: task.status
        }
      };

      return response;
    }
  );

  app.post<{ Params: { taskId: string }; Body: ConfirmCandidateRequest }>(
    "/:taskId/confirm-candidate",
    async (request) => {
      const userId = resolveUserId(request.headers["x-user-id"]);
      const task = await taskService.confirmCandidate(request.params.taskId, request.body, userId);

      const response: ApiResponse<TaskActionResponse> = {
        success: true,
        requestId: request.id,
        data: {
          taskId: task.id,
          status: task.status
        }
      };

      return response;
    }
  );

  app.post<{ Params: { taskId: string }; Body: HumanStepDoneRequest }>(
    "/:taskId/human-step-done",
    async (request) => {
      const userId = resolveUserId(request.headers["x-user-id"]);
      const task = await taskService.markHumanStepDone(request.params.taskId, request.body, userId);

      const response: ApiResponse<TaskActionResponse> = {
        success: true,
        requestId: request.id,
        data: {
          taskId: task.id,
          status: task.status
        }
      };

      return response;
    }
  );

  app.post<{ Params: { taskId: string }; Body: ConfirmSubmitRequest }>(
    "/:taskId/confirm-submit",
    async (request) => {
      const userId = resolveUserId(request.headers["x-user-id"]);
      const task = await taskService.confirmSubmit(request.params.taskId, request.body, userId);

      const response: ApiResponse<TaskActionResponse> = {
        success: true,
        requestId: request.id,
        data: {
          taskId: task.id,
          status: task.status
        }
      };

      return response;
    }
  );

  app.post<{ Params: { taskId: string }; Body: CompletePaymentRequest }>(
    "/:taskId/complete-payment",
    async (request) => {
      const userId = resolveUserId(request.headers["x-user-id"]);
      const task = await taskService.completePayment(request.params.taskId, request.body, userId);

      const response: ApiResponse<TaskActionResponse> = {
        success: true,
        requestId: request.id,
        data: {
          taskId: task.id,
          status: task.status
        }
      };

      return response;
    }
  );

  app.post<{ Params: { taskId: string }; Body: CancelTaskRequest }>(
    "/:taskId/cancel",
    async (request) => {
      const userId = resolveUserId(request.headers["x-user-id"]);
      const task = await taskService.cancelTask(request.params.taskId, request.body, userId);

      const response: ApiResponse<TaskActionResponse> = {
        success: true,
        requestId: request.id,
        data: {
          taskId: task.id,
          status: task.status
        }
      };

      return response;
    }
  );
};

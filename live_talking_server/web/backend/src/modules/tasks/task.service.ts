import { randomUUID } from "node:crypto";

import {
  type BookingIntent,
  type ConversationContext,
  type SeatType,
  type TrainCandidate
} from "../booking/booking.types";
import { browserService } from "../browser/browser.service";
import {
  type BrowserProgressEvent,
  type BrowserActionResult,
  type BrowserSearchTrainsResultData
} from "../browser/browser.types";
import { realtimeService } from "../realtime/realtime.service";
import { AppError } from "../../shared/errors/app-error";
import { ErrorCode } from "../../shared/errors/error-codes";
import { type ID } from "../../shared/types/common";
import {
  AgentTaskStatus,
  BrowserActionType,
  HumanActionType,
  TaskEventType
} from "./task.enums";
import { inMemoryTaskRepository, type TaskRepository } from "./task.repository";
import { canTransition } from "./task.state-machine";
import {
  type CancelTaskRequest,
  type CompletePaymentRequest,
  type ConfirmCandidateRequest,
  type ConfirmIntentRequest,
  type ConfirmSubmitRequest,
  type CreateAssistantTaskRequest,
  type GetTaskEventsResponse,
  type HumanStepDoneRequest,
  type PendingHumanAction,
  type SubmitClarificationsRequest,
  type TaskEntity,
  type TaskEvent
} from "./task.types";

export type TaskTransitionMeta = {
  intent?: BookingIntent;
  context?: ConversationContext;
  selectedCandidateId?: ID | null;
  expiresAt?: string | null;
  terminalReason?: string | null;
  eventMessage?: string;
  eventPayload?: Record<string, unknown>;
  eventType?: TaskEventType;
  requiresAck?: boolean;
  checkpointId?: ID | null;
};

export interface TaskService {
  createTask(input: CreateAssistantTaskRequest, userId: ID): Promise<TaskEntity>;
  getTask(taskId: ID, userId: ID): Promise<TaskEntity>;
  getTaskCandidates(taskId: ID, userId: ID): Promise<TrainCandidate[]>;
  getTaskEvents(taskId: ID, userId: ID, afterSequence?: number): Promise<TaskEvent[]>;
  getTaskEventsResponse(
    taskId: ID,
    userId: ID,
    afterSequence?: number
  ): Promise<GetTaskEventsResponse>;
  getPendingHumanAction(taskId: ID, userId: ID): Promise<PendingHumanAction | null>;
  submitClarifications(
    taskId: ID,
    input: SubmitClarificationsRequest,
    userId: ID
  ): Promise<TaskEntity>;
  confirmIntent(taskId: ID, input: ConfirmIntentRequest, userId: ID): Promise<TaskEntity>;
  confirmCandidate(
    taskId: ID,
    input: ConfirmCandidateRequest,
    userId: ID
  ): Promise<TaskEntity>;
  markHumanStepDone(
    taskId: ID,
    input: HumanStepDoneRequest,
    userId: ID
  ): Promise<TaskEntity>;
  confirmSubmit(taskId: ID, input: ConfirmSubmitRequest, userId: ID): Promise<TaskEntity>;
  completePayment(taskId: ID, input: CompletePaymentRequest, userId: ID): Promise<TaskEntity>;
  cancelTask(taskId: ID, input: CancelTaskRequest, userId: ID): Promise<TaskEntity>;
  appendEvent(taskId: ID, event: TaskEvent): Promise<void>;
  transition(taskId: ID, nextStatus: AgentTaskStatus, meta?: TaskTransitionMeta): Promise<void>;
}

function createEmptyIntent(rawText: string): BookingIntent {
  return {
    rawText,
    travelDate: null,
    origin: null,
    destination: null,
    departureTimeLowerBound: null,
    departureTimeUpperBound: null,
    trainTypes: [],
    seatPreferences: [],
    passengerNames: [],
    allowWaitlist: false,
    allowFallbackTime: false,
    allowFallbackSeat: false,
    queryOnly: false
  };
}

function createInitialContext(taskId: ID): ConversationContext {
  return {
    taskId,
    turns: 0,
    missingSlots: [],
    ambiguousSlots: [],
    slotConfidence: [],
    lastClarificationQuestion: null
  };
}

function getHumanVerificationCheckpointId(taskId: ID): ID {
  return `cp_${taskId}_human_verification`;
}

function getFinalSubmitCheckpointId(taskId: ID): ID {
  return `cp_${taskId}_final_submit`;
}

function getPaymentCheckpointId(taskId: ID): ID {
  return `cp_${taskId}_payment`;
}

function isTerminalStatus(status: AgentTaskStatus): boolean {
  return [
    AgentTaskStatus.COMPLETED,
    AgentTaskStatus.FAILED,
    AgentTaskStatus.CANCELLED,
    AgentTaskStatus.EXPIRED
  ].includes(status);
}

function resolveEventType(from: AgentTaskStatus, to: AgentTaskStatus): TaskEventType {
  if (to === AgentTaskStatus.FAILED) {
    return TaskEventType.TASK_FAILED;
  }

  if (to === AgentTaskStatus.CANCELLED) {
    return TaskEventType.TASK_CANCELLED;
  }

  if (to === AgentTaskStatus.COMPLETED) {
    return TaskEventType.TASK_COMPLETED;
  }

  switch (`${from}->${to}`) {
    case `${AgentTaskStatus.DRAFT}->${AgentTaskStatus.PARSING_INTENT}`:
      return TaskEventType.TASK_CREATED;
    case `${AgentTaskStatus.PARSING_INTENT}->${AgentTaskStatus.AWAITING_SLOT_CLARIFICATION}`:
      return TaskEventType.SLOT_CLARIFICATION_REQUESTED;
    case `${AgentTaskStatus.PARSING_INTENT}->${AgentTaskStatus.AWAITING_INTENT_CONFIRMATION}`:
      return TaskEventType.INTENT_PARSED;
    case `${AgentTaskStatus.AWAITING_INTENT_CONFIRMATION}->${AgentTaskStatus.QUEUED}`:
      return TaskEventType.INTENT_CONFIRMED;
    case `${AgentTaskStatus.QUEUED}->${AgentTaskStatus.SEARCHING}`:
      return TaskEventType.SEARCH_STARTED;
    case `${AgentTaskStatus.SEARCHING}->${AgentTaskStatus.SEARCH_RESULTS_READY}`:
      return TaskEventType.SEARCH_RESULTS_READY;
    case `${AgentTaskStatus.SEARCH_RESULTS_READY}->${AgentTaskStatus.AWAITING_CANDIDATE_CONFIRMATION}`:
      return TaskEventType.SEARCH_RESULTS_READY;
    case `${AgentTaskStatus.AWAITING_CANDIDATE_CONFIRMATION}->${AgentTaskStatus.SELECTING_TRAIN}`:
      return TaskEventType.CANDIDATE_CONFIRMED;
    case `${AgentTaskStatus.SELECTING_TRAIN}->${AgentTaskStatus.LOGIN_REQUIRED}`:
      return TaskEventType.LOGIN_REQUIRED;
    case `${AgentTaskStatus.LOGIN_REQUIRED}->${AgentTaskStatus.AWAITING_HUMAN_VERIFICATION}`:
      return TaskEventType.HUMAN_VERIFICATION_REQUIRED;
    case `${AgentTaskStatus.AWAITING_HUMAN_VERIFICATION}->${AgentTaskStatus.SELECTING_TRAIN}`:
      return TaskEventType.HUMAN_STEP_COMPLETED;
    case `${AgentTaskStatus.AWAITING_FINAL_SUBMIT_CONFIRMATION}->${AgentTaskStatus.SUBMITTING_ORDER}`:
      return TaskEventType.FINAL_SUBMIT_CONFIRMED;
    case `${AgentTaskStatus.SUBMITTING_ORDER}->${AgentTaskStatus.AWAITING_PAYMENT}`:
      return TaskEventType.ORDER_SUBMITTED;
    default:
      return TaskEventType.TASK_CREATED;
  }
}

function resolveRequiresAck(status: AgentTaskStatus): boolean {
  switch (status) {
    case AgentTaskStatus.AWAITING_SLOT_CLARIFICATION:
    case AgentTaskStatus.AWAITING_INTENT_CONFIRMATION:
    case AgentTaskStatus.AWAITING_CANDIDATE_CONFIRMATION:
    case AgentTaskStatus.AWAITING_HUMAN_VERIFICATION:
    case AgentTaskStatus.AWAITING_FINAL_SUBMIT_CONFIRMATION:
    case AgentTaskStatus.AWAITING_PAYMENT:
      return true;
    default:
      return false;
  }
}

function resolveEventMessage(status: AgentTaskStatus): string {
  switch (status) {
    case AgentTaskStatus.PARSING_INTENT:
      return "任务已进入意图解析阶段";
    case AgentTaskStatus.AWAITING_SLOT_CLARIFICATION:
      return "任务等待用户补充关键信息";
    case AgentTaskStatus.AWAITING_INTENT_CONFIRMATION:
      return "任务等待用户确认订票意图";
    case AgentTaskStatus.QUEUED:
      return "任务已进入排队状态";
    case AgentTaskStatus.SEARCHING:
      return "任务开始查询车次";
    case AgentTaskStatus.SEARCH_RESULTS_READY:
      return "查询结果已准备完成";
    case AgentTaskStatus.AWAITING_CANDIDATE_CONFIRMATION:
      return "任务等待用户确认候选车次";
    case AgentTaskStatus.SELECTING_TRAIN:
      return "任务开始尝试选择车次与席位";
    case AgentTaskStatus.LOGIN_REQUIRED:
      return "任务需要先完成登录";
    case AgentTaskStatus.AWAITING_HUMAN_VERIFICATION:
      return "任务等待人工完成验证步骤";
    case AgentTaskStatus.AWAITING_FINAL_SUBMIT_CONFIRMATION:
      return "任务等待用户进行最终提交确认";
    case AgentTaskStatus.AWAITING_PAYMENT:
      return "订单已提交，等待支付";
    case AgentTaskStatus.COMPLETED:
      return "任务已完成";
    case AgentTaskStatus.FAILED:
      return "任务执行失败";
    case AgentTaskStatus.CANCELLED:
      return "任务已取消";
    default:
      return `任务状态已切换为 ${status}`;
  }
}

function extractCandidates(result: BrowserActionResult): TrainCandidate[] {
  const data = result.data as BrowserSearchTrainsResultData | undefined;

  if (!data || !Array.isArray(data.candidates)) {
    return [];
  }

  return data.candidates;
}

function pickAutoSelectedCandidate(candidates: TrainCandidate[]): TrainCandidate | null {
  if (candidates.length === 0) {
    return null;
  }

  return (
    candidates.find((candidate) => candidate.isRecommended) ??
    [...candidates].sort((left, right) => right.score - left.score)[0] ??
    candidates[0]
  );
}

function pickAutoSelectedSeatType(intent: BookingIntent, candidate: TrainCandidate): SeatType {
  for (const preferredSeatType of intent.seatPreferences) {
    const matchedInventory = candidate.seatInventory.find(
      (inventory) =>
        inventory.seatType === preferredSeatType &&
        inventory.normalizedStatus !== "无" &&
        inventory.normalizedStatus !== "未知"
    );

    if (matchedInventory) {
      return preferredSeatType;
    }
  }

  const firstAvailableSeat = candidate.seatInventory.find(
    (inventory) => inventory.normalizedStatus !== "无" && inventory.normalizedStatus !== "未知"
  );

  return firstAvailableSeat?.seatType ?? intent.seatPreferences[0] ?? "二等座";
}

export class InMemoryTaskService implements TaskService {
  private readonly activeQueuedWorkers = new Set<ID>();
  private readonly activeSelectWorkers = new Set<ID>();
  private readonly activeSubmitWorkers = new Set<ID>();

  constructor(private readonly repository: TaskRepository = inMemoryTaskRepository) {}

  async createTask(input: CreateAssistantTaskRequest, userId: ID): Promise<TaskEntity> {
    const now = new Date().toISOString();
    const taskId = `task_${randomUUID()}`;
    const task: TaskEntity = {
      id: taskId,
      userId,
      taskType: input.taskType,
      status: AgentTaskStatus.DRAFT,
      rawText: input.rawText,
      intent: createEmptyIntent(input.rawText),
      context: createInitialContext(taskId),
      selectedCandidateId: null,
      createdAt: now,
      updatedAt: now,
      expiresAt: null,
      terminalReason: null
    };

    await this.repository.createTask(task);

    await this.appendEvent(task.id, {
      id: `evt_${randomUUID()}`,
      taskId: task.id,
      sequenceId: await this.repository.nextSequenceId(task.id),
      type: TaskEventType.TASK_CREATED,
      status: task.status,
      checkpointId: null,
      message: "任务已创建",
      payload: {
        taskType: task.taskType,
        rawText: task.rawText
      },
      requiresAck: false,
      createdAt: now
    });

    return task;
  }

  async getTask(taskId: ID, userId: ID): Promise<TaskEntity> {
    const task = await this.loadTask(taskId);

    if (task.userId !== userId) {
      throw new AppError(404, ErrorCode.TASK_NOT_FOUND, "Task not found.");
    }

    return task;
  }

  async getTaskCandidates(taskId: ID, userId: ID): Promise<TrainCandidate[]> {
    await this.getTask(taskId, userId);

    return this.repository.getCandidates(taskId);
  }

  async getTaskEvents(taskId: ID, userId: ID, afterSequence = 0): Promise<TaskEvent[]> {
    await this.getTask(taskId, userId);

    return this.repository.listEvents(taskId, afterSequence);
  }

  async getTaskEventsResponse(
    taskId: ID,
    userId: ID,
    afterSequence = 0
  ): Promise<GetTaskEventsResponse> {
    const events = await this.getTaskEvents(taskId, userId, afterSequence);
    const nextSequenceId = await this.getNextSequenceId(taskId, userId);

    return {
      taskId,
      events,
      nextSequenceId
    };
  }

  async getPendingHumanAction(taskId: ID, userId: ID): Promise<PendingHumanAction | null> {
    const task = await this.getTask(taskId, userId);

    switch (task.status) {
      case AgentTaskStatus.AWAITING_SLOT_CLARIFICATION:
        return {
          type: HumanActionType.CLARIFY_SLOTS,
          checkpointId: `cp_${task.id}_clarify`,
          prompt: "请补充缺失或存在歧义的订票信息。"
        };
      case AgentTaskStatus.AWAITING_INTENT_CONFIRMATION:
        return {
          type: HumanActionType.CONFIRM_INTENT,
          checkpointId: `cp_${task.id}_intent_confirm`,
          prompt: "请确认当前解析出的订票意图。"
        };
      case AgentTaskStatus.AWAITING_CANDIDATE_CONFIRMATION:
        return {
          type: HumanActionType.CONFIRM_CANDIDATE,
          checkpointId: `cp_${task.id}_candidate_confirm`,
          prompt: "请选择并确认目标车次。"
        };
      case AgentTaskStatus.AWAITING_HUMAN_VERIFICATION:
        return {
          type: HumanActionType.COMPLETE_QR_LOGIN,
          checkpointId: getHumanVerificationCheckpointId(task.id),
          prompt: "请在 12306 页面完成扫码登录或验证码验证。"
        };
      case AgentTaskStatus.AWAITING_FINAL_SUBMIT_CONFIRMATION:
        return {
          type: HumanActionType.CONFIRM_FINAL_SUBMIT,
          checkpointId: getFinalSubmitCheckpointId(task.id),
          prompt: "请确认是否提交订单。"
        };
      case AgentTaskStatus.AWAITING_PAYMENT:
        return {
          type: HumanActionType.COMPLETE_PAYMENT,
          checkpointId: `cp_${task.id}_payment`,
          prompt: "请在 12306 页面完成支付。"
        };
      default:
        return null;
    }
  }

  async submitClarifications(): Promise<TaskEntity> {
    throw new AppError(501, ErrorCode.INTERNAL_ERROR, "submitClarifications is not implemented yet.");
  }

  async confirmIntent(
    taskId: ID,
    input: ConfirmIntentRequest,
    userId: ID
  ): Promise<TaskEntity> {
    const task = await this.getTask(taskId, userId);

    if (input.confirmed) {
      await this.transition(task.id, AgentTaskStatus.QUEUED, {
        eventMessage: "用户已确认订票意图，任务进入排队状态",
        eventPayload: {
          confirmed: true
        }
      });
    } else {
      await this.transition(task.id, AgentTaskStatus.CANCELLED, {
        terminalReason: "intent_rejected",
        eventMessage: "用户拒绝当前订票意图，任务已取消",
        eventPayload: {
          confirmed: false
        }
      });
    }

    return this.getTask(task.id, userId);
  }

  async confirmCandidate(
    taskId: ID,
    input: ConfirmCandidateRequest,
    userId: ID,
    options: {
      autoConfirmed?: boolean;
    } = {}
  ): Promise<TaskEntity> {
    const task = await this.getTask(taskId, userId);
    const candidates = await this.repository.getCandidates(taskId);
    const matchedCandidate = candidates.find((candidate) => candidate.trainNo === input.candidateTrainNo);
    const autoConfirmed = options.autoConfirmed ?? false;

    if (!matchedCandidate) {
      throw new AppError(404, ErrorCode.NO_TRAIN_MATCHED, "Candidate train not found.");
    }

    if (
      task.status !== AgentTaskStatus.SEARCH_RESULTS_READY &&
      task.status !== AgentTaskStatus.AWAITING_CANDIDATE_CONFIRMATION
    ) {
      throw new AppError(
        409,
        ErrorCode.TASK_STATUS_INVALID,
        `Task is not ready for candidate confirmation: ${task.status}.`
      );
    }

    if (task.status === AgentTaskStatus.SEARCH_RESULTS_READY) {
      await this.transition(taskId, AgentTaskStatus.AWAITING_CANDIDATE_CONFIRMATION, {
        eventMessage: autoConfirmed
          ? `系统已自动锁定推荐车次 ${input.candidateTrainNo}`
          : "用户开始确认候选车次",
        eventPayload: {
          candidateTrainNo: input.candidateTrainNo,
          seatType: input.seatType,
          allowWaitlist: input.allowWaitlist,
          autoConfirmed
        },
        requiresAck: autoConfirmed ? false : undefined
      });
    }

    const latestTask = await this.getTask(taskId, userId);
    const nextIntent: BookingIntent = {
      ...latestTask.intent,
      seatPreferences: [
        input.seatType,
        ...latestTask.intent.seatPreferences.filter((seatType) => seatType !== input.seatType)
      ],
      allowWaitlist: input.allowWaitlist
    };

    await this.transition(taskId, AgentTaskStatus.SELECTING_TRAIN, {
      selectedCandidateId: input.candidateTrainNo,
      intent: nextIntent,
      eventType: TaskEventType.CANDIDATE_CONFIRMED,
      eventMessage: autoConfirmed
        ? `已自动确认推荐车次 ${input.candidateTrainNo}，开始尝试点击“预订”`
        : `已确认车次 ${input.candidateTrainNo}，开始尝试选座`,
      eventPayload: {
        candidateTrainNo: input.candidateTrainNo,
        seatType: input.seatType,
        allowWaitlist: input.allowWaitlist,
        autoConfirmed
      }
    });

    this.scheduleSelectTrain(taskId);

    return this.getTask(taskId, userId);
  }

  async markHumanStepDone(
    taskId: ID,
    input: HumanStepDoneRequest,
    userId: ID
  ): Promise<TaskEntity> {
    const task = await this.getTask(taskId, userId);
    const expectedCheckpointId = getHumanVerificationCheckpointId(taskId);
    const allowedActions = new Set<HumanActionType>([
      HumanActionType.COMPLETE_QR_LOGIN,
      HumanActionType.COMPLETE_CAPTCHA,
      HumanActionType.COMPLETE_SMS_VERIFICATION
    ]);

    if (task.status !== AgentTaskStatus.AWAITING_HUMAN_VERIFICATION) {
      throw new AppError(
        409,
        ErrorCode.TASK_STATUS_INVALID,
        `Task is not awaiting human verification: ${task.status}.`
      );
    }

    if (input.checkpointId !== expectedCheckpointId) {
      throw new AppError(409, ErrorCode.TASK_STATUS_INVALID, "Checkpoint ID does not match pending human action.", {
        expectedCheckpointId,
        actualCheckpointId: input.checkpointId
      });
    }

    if (!allowedActions.has(input.actionType)) {
      throw new AppError(409, ErrorCode.TASK_STATUS_INVALID, "Unsupported human action type for this step.", {
        actionType: input.actionType
      });
    }

    browserService.markHumanVerified(taskId);

    await this.transition(taskId, AgentTaskStatus.SELECTING_TRAIN, {
      eventType: TaskEventType.HUMAN_STEP_COMPLETED,
      eventMessage: "人工验证已完成，恢复选座流程",
      eventPayload: {
        checkpointId: input.checkpointId,
        actionType: input.actionType
      }
    });

    this.scheduleSelectTrain(taskId);

    return this.getTask(taskId, userId);
  }

  async confirmSubmit(
    taskId: ID,
    input: ConfirmSubmitRequest,
    userId: ID
  ): Promise<TaskEntity> {
    const task = await this.getTask(taskId, userId);
    const expectedCheckpointId = getFinalSubmitCheckpointId(taskId);

    if (task.status !== AgentTaskStatus.AWAITING_FINAL_SUBMIT_CONFIRMATION) {
      throw new AppError(
        409,
        ErrorCode.TASK_STATUS_INVALID,
        `Task is not awaiting final submit confirmation: ${task.status}.`
      );
    }

    if (input.checkpointId !== expectedCheckpointId) {
      throw new AppError(409, ErrorCode.TASK_STATUS_INVALID, "Checkpoint ID does not match pending submit step.", {
        expectedCheckpointId,
        actualCheckpointId: input.checkpointId
      });
    }

    if (input.confirmed) {
      await this.transition(taskId, AgentTaskStatus.SUBMITTING_ORDER, {
        checkpointId: expectedCheckpointId,
        eventType: TaskEventType.FINAL_SUBMIT_CONFIRMED,
        eventMessage: "用户已确认最终提交订单，开始执行提交流程",
        eventPayload: {
          checkpointId: input.checkpointId,
          confirmed: true
        }
      });
    } else {
      await this.transition(taskId, AgentTaskStatus.CANCELLED, {
        terminalReason: "final_submit_rejected",
        eventMessage: "用户取消了最终提交，任务已终止",
        eventPayload: {
          checkpointId: input.checkpointId,
          confirmed: false
        }
      });
    }

    return this.getTask(taskId, userId);
  }

  async completePayment(
    taskId: ID,
    input: CompletePaymentRequest,
    userId: ID
  ): Promise<TaskEntity> {
    const task = await this.getTask(taskId, userId);
    const expectedCheckpointId = getPaymentCheckpointId(taskId);

    if (task.status !== AgentTaskStatus.AWAITING_PAYMENT) {
      throw new AppError(
        409,
        ErrorCode.TASK_STATUS_INVALID,
        `Task is not awaiting payment: ${task.status}.`
      );
    }

    if (input.checkpointId !== expectedCheckpointId) {
      throw new AppError(409, ErrorCode.TASK_STATUS_INVALID, "Checkpoint ID does not match payment step.", {
        expectedCheckpointId,
        actualCheckpointId: input.checkpointId
      });
    }

    await this.transition(taskId, AgentTaskStatus.COMPLETED, {
      checkpointId: expectedCheckpointId,
      eventMessage: "支付成功，祝您旅途愉快！",
      eventPayload: {
        checkpointId: input.checkpointId,
        actionType: HumanActionType.COMPLETE_PAYMENT
      },
      requiresAck: false
    });

    return this.getTask(taskId, userId);
  }

  async cancelTask(taskId: ID, input: CancelTaskRequest, userId: ID): Promise<TaskEntity> {
    const task = await this.getTask(taskId, userId);

    if (isTerminalStatus(task.status)) {
      throw new AppError(
        409,
        ErrorCode.TASK_STATUS_INVALID,
        `Task is already terminal and cannot be cancelled: ${task.status}.`
      );
    }

    await this.transition(taskId, AgentTaskStatus.CANCELLED, {
      terminalReason: input.reason ?? "cancelled_by_user",
      eventMessage: "用户已取消当前任务",
      eventPayload: {
        reason: input.reason ?? null
      }
    });

    return this.getTask(taskId, userId);
  }

  async appendEvent(taskId: ID, event: TaskEvent): Promise<void> {
    await this.loadTask(taskId);
    await this.repository.appendEvent(event);
  }

  async transition(taskId: ID, nextStatus: AgentTaskStatus, meta: TaskTransitionMeta = {}): Promise<void> {
    const currentTask = await this.loadTask(taskId);

    if (!canTransition(currentTask.status, nextStatus)) {
      throw new AppError(
        409,
        ErrorCode.TASK_STATUS_INVALID,
        `Illegal task transition: ${currentTask.status} -> ${nextStatus}.`,
        {
          from: currentTask.status,
          to: nextStatus
        }
      );
    }

    const now = new Date().toISOString();
    const updatedTask: TaskEntity = {
      ...currentTask,
      status: nextStatus,
      intent: meta.intent ?? currentTask.intent,
      context: meta.context ?? currentTask.context,
      selectedCandidateId:
        meta.selectedCandidateId !== undefined
          ? meta.selectedCandidateId
          : currentTask.selectedCandidateId,
      expiresAt: meta.expiresAt !== undefined ? meta.expiresAt : currentTask.expiresAt,
      terminalReason:
        meta.terminalReason !== undefined ? meta.terminalReason : currentTask.terminalReason,
      updatedAt: now
    };

    await this.repository.updateTask(updatedTask);

    const event: TaskEvent = {
      id: `evt_${randomUUID()}`,
      taskId,
      sequenceId: await this.repository.nextSequenceId(taskId),
      type: meta.eventType ?? resolveEventType(currentTask.status, nextStatus),
      status: nextStatus,
      checkpointId: meta.checkpointId ?? null,
      message: meta.eventMessage ?? resolveEventMessage(nextStatus),
      payload: meta.eventPayload ?? {},
      requiresAck: meta.requiresAck ?? resolveRequiresAck(nextStatus),
      createdAt: now
    };

    await this.appendEvent(taskId, event);
    await realtimeService.publishTaskEvent(taskId, event);

    if (nextStatus === AgentTaskStatus.QUEUED) {
      this.scheduleQueuedSearch(taskId);
    }

    if (nextStatus === AgentTaskStatus.SUBMITTING_ORDER) {
      this.scheduleSubmitOrder(taskId);
    }
  }

  private createBrowserProgressReporter(taskId: ID) {
    return async (progress: BrowserProgressEvent): Promise<void> => {
      const task = await this.repository.getTask(taskId);
      if (!task) {
        return;
      }

      const now = new Date().toISOString();
      const checkpointId =
        typeof progress.payload?.checkpointId === "string"
          ? (progress.payload.checkpointId as ID)
          : null;

      const event: TaskEvent = {
        id: `evt_${randomUUID()}`,
        taskId,
        sequenceId: await this.repository.nextSequenceId(taskId),
        type: TaskEventType.BROWSER_PROGRESS,
        status: task.status,
        checkpointId,
        message: progress.message,
        payload: {
          logLevel: progress.level,
          ...(progress.payload ?? {})
        },
        requiresAck: false,
        createdAt: now
      };

      await this.appendEvent(taskId, event);
      await realtimeService.publishTaskEvent(taskId, event);
    };
  }

  private async getNextSequenceId(taskId: ID, userId: ID): Promise<number> {
    await this.getTask(taskId, userId);

    return this.repository.nextSequenceId(taskId);
  }

  private async loadTask(taskId: ID): Promise<TaskEntity> {
    const task = await this.repository.getTask(taskId);

    if (!task) {
      throw new AppError(404, ErrorCode.TASK_NOT_FOUND, "Task not found.");
    }

    return task;
  }

  private scheduleQueuedSearch(taskId: ID): void {
    if (this.activeQueuedWorkers.has(taskId)) {
      return;
    }

    this.activeQueuedWorkers.add(taskId);
    setTimeout(() => {
      void this.runQueuedSearch(taskId).finally(() => {
        this.activeQueuedWorkers.delete(taskId);
      });
    }, 0);
  }

  private scheduleSelectTrain(taskId: ID): void {
    if (this.activeSelectWorkers.has(taskId)) {
      return;
    }

    this.activeSelectWorkers.add(taskId);
    setTimeout(() => {
      void this.runSelectTrain(taskId).finally(() => {
        this.activeSelectWorkers.delete(taskId);
      });
    }, 0);
  }

  private scheduleSubmitOrder(taskId: ID): void {
    if (this.activeSubmitWorkers.has(taskId)) {
      return;
    }

    this.activeSubmitWorkers.add(taskId);
    setTimeout(() => {
      void this.runSubmitOrder(taskId).finally(() => {
        this.activeSubmitWorkers.delete(taskId);
      });
    }, 0);
  }

  private async runQueuedSearch(taskId: ID): Promise<void> {
    try {
      const currentTask = await this.loadTask(taskId);
      if (currentTask.status !== AgentTaskStatus.QUEUED) {
        return;
      }

      await this.transition(taskId, AgentTaskStatus.SEARCHING, {
        eventMessage: "任务已启动浏览器查询流程"
      });

      const latestTask = await this.loadTask(taskId);
      const onProgress = this.createBrowserProgressReporter(taskId);
      const result = await browserService.execute({
        action: BrowserActionType.SEARCH_TRAINS,
        taskId,
        payload: {
          intent: latestTask.intent
        },
        onProgress
      });

      await this.handleBrowserResult(taskId, result);
    } catch (error) {
      await this.failTaskIfPossible(taskId, error, "browser_worker_failed", "浏览器查询流程失败。");
    }
  }

  private async runSelectTrain(taskId: ID): Promise<void> {
    try {
      const currentTask = await this.loadTask(taskId);
      if (currentTask.status !== AgentTaskStatus.SELECTING_TRAIN) {
        return;
      }

      if (!currentTask.selectedCandidateId) {
        throw new AppError(409, ErrorCode.NO_TRAIN_MATCHED, "No candidate selected for train selection.");
      }

      const shouldResumeAfterLogin = browserService.hasPendingLoginSession(taskId);
      const onProgress = this.createBrowserProgressReporter(taskId);

      const result = await browserService.execute(
        shouldResumeAfterLogin
          ? {
              action: BrowserActionType.WAIT_LOGIN,
              taskId,
              payload: {
                timeoutMs: 120000
              },
              onProgress
            }
          : {
              action: BrowserActionType.SELECT_TRAIN,
              taskId,
              payload: {
                intent: currentTask.intent,
                candidateTrainNo: currentTask.selectedCandidateId,
                seatType: currentTask.intent.seatPreferences[0] ?? "二等座",
                passengerNames: currentTask.intent.passengerNames
              },
              onProgress
            }
      );

      await this.handleBrowserResult(taskId, result);
    } catch (error) {
      await this.failTaskIfPossible(taskId, error, "browser_select_failed", "Browser select flow failed.");
    }
  }

  private async runSubmitOrder(taskId: ID): Promise<void> {
    try {
      const currentTask = await this.loadTask(taskId);
      if (currentTask.status !== AgentTaskStatus.SUBMITTING_ORDER) {
        return;
      }

      const onProgress = this.createBrowserProgressReporter(taskId);
      const result = await browserService.execute({
        action: BrowserActionType.SUBMIT_ORDER,
        taskId,
        payload: {},
        onProgress
      });

      await this.handleBrowserResult(taskId, result);
    } catch (error) {
      await this.failTaskIfPossible(taskId, error, "browser_submit_failed", "Browser submit flow failed.");
    }
  }

  private async handleBrowserResult(taskId: ID, result: BrowserActionResult): Promise<void> {
    if (result.ok && result.state === "search_results_ready") {
      const candidates = extractCandidates(result);
      await this.repository.saveCandidates(taskId, candidates);
      await this.transition(taskId, AgentTaskStatus.SEARCH_RESULTS_READY, {
        eventMessage: `浏览器查询完成，返回 ${candidates.length} 条候选车次`,
        eventPayload: {
          candidateCount: candidates.length
        }
      });

      const currentTask = await this.loadTask(taskId);
      const autoSelectedCandidate = pickAutoSelectedCandidate(candidates);

      if (!autoSelectedCandidate) {
        return;
      }

      await this.confirmCandidate(
        taskId,
        {
          idempotencyKey: `auto_confirm_candidate_${randomUUID()}`,
          candidateTrainNo: autoSelectedCandidate.trainNo,
          seatType: pickAutoSelectedSeatType(currentTask.intent, autoSelectedCandidate),
          allowWaitlist: currentTask.intent.allowWaitlist
        },
        currentTask.userId,
        {
          autoConfirmed: true
        }
      );
      return;
    }

    if (result.state === "login_required") {
      await this.transition(taskId, AgentTaskStatus.LOGIN_REQUIRED, {
        eventMessage: "浏览器流程检测到需要登录",
        eventPayload: result.data,
        eventType: TaskEventType.LOGIN_REQUIRED
      });
      await this.transition(taskId, AgentTaskStatus.AWAITING_HUMAN_VERIFICATION, {
        checkpointId: getHumanVerificationCheckpointId(taskId),
        eventType: TaskEventType.HUMAN_VERIFICATION_REQUIRED,
        eventMessage: "请扫码登录后点击“人工步骤完成”继续",
        eventPayload: {
          ...(result.data ?? {}),
          checkpointId: getHumanVerificationCheckpointId(taskId)
        },
        requiresAck: true
      });
      return;
    }

    if (result.state === "waiting_human") {
      await this.transition(taskId, AgentTaskStatus.AWAITING_HUMAN_VERIFICATION, {
        checkpointId: getHumanVerificationCheckpointId(taskId),
        eventType: TaskEventType.HUMAN_VERIFICATION_REQUIRED,
        eventMessage: "浏览器流程仍需要人工继续操作",
        eventPayload: {
          ...(result.data ?? {}),
          checkpointId: getHumanVerificationCheckpointId(taskId)
        },
        requiresAck: true
      });
      return;
    }

    if (result.ok && (result.state === "order_page_ready" || result.state === "train_selected")) {
      await this.transition(taskId, AgentTaskStatus.AWAITING_FINAL_SUBMIT_CONFIRMATION, {
        checkpointId: getFinalSubmitCheckpointId(taskId),
        eventType: TaskEventType.HUMAN_STEP_COMPLETED,
        eventMessage: "选座成功，系统将自动提交订单",
        eventPayload: {
          ...(result.data ?? {}),
          checkpointId: getFinalSubmitCheckpointId(taskId),
          autoSubmit: true
        },
        requiresAck: false
      });
      await this.transition(taskId, AgentTaskStatus.SUBMITTING_ORDER, {
        checkpointId: getFinalSubmitCheckpointId(taskId),
        eventType: TaskEventType.FINAL_SUBMIT_CONFIRMED,
        eventMessage: "系统已自动确认最终提交，开始执行提交流程",
        eventPayload: {
          ...(result.data ?? {}),
          checkpointId: getFinalSubmitCheckpointId(taskId),
          confirmed: true,
          autoSubmit: true
        },
        requiresAck: false
      });
      return;
    }

    if (result.ok && result.state === "order_submitted") {
      await this.transition(taskId, AgentTaskStatus.AWAITING_PAYMENT, {
        checkpointId: getPaymentCheckpointId(taskId),
        eventType: TaskEventType.ORDER_SUBMITTED,
        eventMessage: "订单已提交，等待用户完成支付",
        eventPayload: {
          ...(result.data ?? {}),
          checkpointId: getPaymentCheckpointId(taskId)
        },
        requiresAck: true
      });
      return;
    }

    throw new AppError(
      500,
      result.errorCode ?? ErrorCode.INTERNAL_ERROR,
      result.errorMessage ?? "Browser action failed."
    );
  }

  private async failTaskIfPossible(
    taskId: ID,
    error: unknown,
    terminalReason: string,
    fallbackMessage: string
  ): Promise<void> {
    const latestTask = await this.repository.getTask(taskId);
    if (
      !latestTask ||
      [AgentTaskStatus.FAILED, AgentTaskStatus.CANCELLED, AgentTaskStatus.EXPIRED].includes(latestTask.status)
    ) {
      return;
    }

    const message = error instanceof Error ? error.message : fallbackMessage;
    const code = error instanceof AppError ? error.code : ErrorCode.INTERNAL_ERROR;

    await this.transition(taskId, AgentTaskStatus.FAILED, {
      terminalReason,
      eventMessage: message,
      eventPayload: {
        code
      }
    });
  }
}

export const taskService = new InMemoryTaskService();



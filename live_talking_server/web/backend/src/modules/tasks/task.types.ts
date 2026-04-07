import { type ErrorCode } from "../../shared/errors/error-codes";
import { type ID, type ISODateTime, type Nullable } from "../../shared/types/common";
import {
  AgentTaskStatus,
  CheckpointStatus,
  HumanActionType,
  TaskEventType
} from "./task.enums";
import {
  type BookingIntent,
  type ConversationContext,
  type SeatType,
  type TrainCandidate
} from "../booking/booking.types";

export type TaskEntity = {
  id: ID;
  userId: ID;
  taskType: "book_train_ticket";
  status: AgentTaskStatus;
  rawText: string;
  intent: BookingIntent;
  context: ConversationContext;
  selectedCandidateId: Nullable<ID>;
  createdAt: ISODateTime;
  updatedAt: ISODateTime;
  expiresAt: Nullable<ISODateTime>;
  terminalReason: Nullable<string>;
};

export type TaskCheckpoint = {
  id: ID;
  taskId: ID;
  name: string;
  status: CheckpointStatus;
  inputSummary: Record<string, unknown>;
  outputSummary: Record<string, unknown>;
  retryCount: number;
  maxRetryCount: number;
  manualTakeoverAllowed: boolean;
  screenshotUrl: Nullable<string>;
  traceId: Nullable<string>;
  errorCode: Nullable<ErrorCode>;
  errorMessage: Nullable<string>;
  createdAt: ISODateTime;
  updatedAt: ISODateTime;
};

export type TaskEvent = {
  id: ID;
  taskId: ID;
  sequenceId: number;
  type: TaskEventType;
  status: AgentTaskStatus;
  checkpointId: Nullable<ID>;
  message: string;
  payload: Record<string, unknown>;
  requiresAck: boolean;
  createdAt: ISODateTime;
};

export type PendingHumanAction = {
  type: HumanActionType;
  checkpointId: ID;
  prompt: string;
};

export type CreateAssistantTaskRequest = {
  taskType: "book_train_ticket";
  rawText: string;
  idempotencyKey: string;
};

export type GetTaskDetailResponse = {
  task: TaskEntity;
  checkpoints: TaskCheckpoint[];
  recentEvents: TaskEvent[];
  candidates: TrainCandidate[];
  pendingHumanAction: Nullable<PendingHumanAction>;
};

export type SubmitClarificationsRequest = {
  idempotencyKey: string;
  clarifications: Partial<BookingIntent>;
};

export type ConfirmIntentRequest = {
  idempotencyKey: string;
  confirmed: boolean;
};

export type GetCandidatesResponse = {
  taskId: ID;
  status: AgentTaskStatus;
  candidates: TrainCandidate[];
};

export type ConfirmCandidateRequest = {
  idempotencyKey: string;
  candidateTrainNo: string;
  seatType: SeatType;
  allowWaitlist: boolean;
};

export type HumanStepDoneRequest = {
  idempotencyKey: string;
  checkpointId: ID;
  actionType: HumanActionType;
};

export type ConfirmSubmitRequest = {
  idempotencyKey: string;
  checkpointId: ID;
  confirmed: boolean;
};

export type CompletePaymentRequest = {
  idempotencyKey: string;
  checkpointId: ID;
};

export type CancelTaskRequest = {
  idempotencyKey: string;
  reason?: string;
};

export type GetTaskEventsResponse = {
  taskId: ID;
  events: TaskEvent[];
  nextSequenceId: number;
};

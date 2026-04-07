"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.taskService = exports.InMemoryTaskService = void 0;
const node_crypto_1 = require("node:crypto");
const browser_service_1 = require("../browser/browser.service");
const realtime_service_1 = require("../realtime/realtime.service");
const app_error_1 = require("../../shared/errors/app-error");
const error_codes_1 = require("../../shared/errors/error-codes");
const task_enums_1 = require("./task.enums");
const task_repository_1 = require("./task.repository");
const task_state_machine_1 = require("./task.state-machine");
function createEmptyIntent(rawText) {
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
function createInitialContext(taskId) {
    return {
        taskId,
        turns: 0,
        missingSlots: [],
        ambiguousSlots: [],
        slotConfidence: [],
        lastClarificationQuestion: null
    };
}
function getHumanVerificationCheckpointId(taskId) {
    return `cp_${taskId}_human_verification`;
}
function getFinalSubmitCheckpointId(taskId) {
    return `cp_${taskId}_final_submit`;
}
function getPaymentCheckpointId(taskId) {
    return `cp_${taskId}_payment`;
}
function isTerminalStatus(status) {
    return [
        task_enums_1.AgentTaskStatus.COMPLETED,
        task_enums_1.AgentTaskStatus.FAILED,
        task_enums_1.AgentTaskStatus.CANCELLED,
        task_enums_1.AgentTaskStatus.EXPIRED
    ].includes(status);
}
function resolveEventType(from, to) {
    if (to === task_enums_1.AgentTaskStatus.FAILED) {
        return task_enums_1.TaskEventType.TASK_FAILED;
    }
    if (to === task_enums_1.AgentTaskStatus.CANCELLED) {
        return task_enums_1.TaskEventType.TASK_CANCELLED;
    }
    if (to === task_enums_1.AgentTaskStatus.COMPLETED) {
        return task_enums_1.TaskEventType.TASK_COMPLETED;
    }
    switch (`${from}->${to}`) {
        case `${task_enums_1.AgentTaskStatus.DRAFT}->${task_enums_1.AgentTaskStatus.PARSING_INTENT}`:
            return task_enums_1.TaskEventType.TASK_CREATED;
        case `${task_enums_1.AgentTaskStatus.PARSING_INTENT}->${task_enums_1.AgentTaskStatus.AWAITING_SLOT_CLARIFICATION}`:
            return task_enums_1.TaskEventType.SLOT_CLARIFICATION_REQUESTED;
        case `${task_enums_1.AgentTaskStatus.PARSING_INTENT}->${task_enums_1.AgentTaskStatus.AWAITING_INTENT_CONFIRMATION}`:
            return task_enums_1.TaskEventType.INTENT_PARSED;
        case `${task_enums_1.AgentTaskStatus.AWAITING_INTENT_CONFIRMATION}->${task_enums_1.AgentTaskStatus.QUEUED}`:
            return task_enums_1.TaskEventType.INTENT_CONFIRMED;
        case `${task_enums_1.AgentTaskStatus.QUEUED}->${task_enums_1.AgentTaskStatus.SEARCHING}`:
            return task_enums_1.TaskEventType.SEARCH_STARTED;
        case `${task_enums_1.AgentTaskStatus.SEARCHING}->${task_enums_1.AgentTaskStatus.SEARCH_RESULTS_READY}`:
            return task_enums_1.TaskEventType.SEARCH_RESULTS_READY;
        case `${task_enums_1.AgentTaskStatus.SEARCH_RESULTS_READY}->${task_enums_1.AgentTaskStatus.AWAITING_CANDIDATE_CONFIRMATION}`:
            return task_enums_1.TaskEventType.SEARCH_RESULTS_READY;
        case `${task_enums_1.AgentTaskStatus.AWAITING_CANDIDATE_CONFIRMATION}->${task_enums_1.AgentTaskStatus.SELECTING_TRAIN}`:
            return task_enums_1.TaskEventType.CANDIDATE_CONFIRMED;
        case `${task_enums_1.AgentTaskStatus.SELECTING_TRAIN}->${task_enums_1.AgentTaskStatus.LOGIN_REQUIRED}`:
            return task_enums_1.TaskEventType.LOGIN_REQUIRED;
        case `${task_enums_1.AgentTaskStatus.LOGIN_REQUIRED}->${task_enums_1.AgentTaskStatus.AWAITING_HUMAN_VERIFICATION}`:
            return task_enums_1.TaskEventType.HUMAN_VERIFICATION_REQUIRED;
        case `${task_enums_1.AgentTaskStatus.AWAITING_HUMAN_VERIFICATION}->${task_enums_1.AgentTaskStatus.SELECTING_TRAIN}`:
            return task_enums_1.TaskEventType.HUMAN_STEP_COMPLETED;
        case `${task_enums_1.AgentTaskStatus.AWAITING_FINAL_SUBMIT_CONFIRMATION}->${task_enums_1.AgentTaskStatus.SUBMITTING_ORDER}`:
            return task_enums_1.TaskEventType.FINAL_SUBMIT_CONFIRMED;
        case `${task_enums_1.AgentTaskStatus.SUBMITTING_ORDER}->${task_enums_1.AgentTaskStatus.AWAITING_PAYMENT}`:
            return task_enums_1.TaskEventType.ORDER_SUBMITTED;
        default:
            return task_enums_1.TaskEventType.TASK_CREATED;
    }
}
function resolveRequiresAck(status) {
    switch (status) {
        case task_enums_1.AgentTaskStatus.AWAITING_SLOT_CLARIFICATION:
        case task_enums_1.AgentTaskStatus.AWAITING_INTENT_CONFIRMATION:
        case task_enums_1.AgentTaskStatus.AWAITING_CANDIDATE_CONFIRMATION:
        case task_enums_1.AgentTaskStatus.AWAITING_HUMAN_VERIFICATION:
        case task_enums_1.AgentTaskStatus.AWAITING_FINAL_SUBMIT_CONFIRMATION:
        case task_enums_1.AgentTaskStatus.AWAITING_PAYMENT:
            return true;
        default:
            return false;
    }
}
function resolveEventMessage(status) {
    switch (status) {
        case task_enums_1.AgentTaskStatus.PARSING_INTENT:
            return "任务已进入意图解析阶段";
        case task_enums_1.AgentTaskStatus.AWAITING_SLOT_CLARIFICATION:
            return "任务等待用户补充关键信息";
        case task_enums_1.AgentTaskStatus.AWAITING_INTENT_CONFIRMATION:
            return "任务等待用户确认订票意图";
        case task_enums_1.AgentTaskStatus.QUEUED:
            return "任务已进入排队状态";
        case task_enums_1.AgentTaskStatus.SEARCHING:
            return "任务开始查询车次";
        case task_enums_1.AgentTaskStatus.SEARCH_RESULTS_READY:
            return "查询结果已准备完成";
        case task_enums_1.AgentTaskStatus.AWAITING_CANDIDATE_CONFIRMATION:
            return "任务等待用户确认候选车次";
        case task_enums_1.AgentTaskStatus.SELECTING_TRAIN:
            return "任务开始尝试选择车次与席位";
        case task_enums_1.AgentTaskStatus.LOGIN_REQUIRED:
            return "任务需要先完成登录";
        case task_enums_1.AgentTaskStatus.AWAITING_HUMAN_VERIFICATION:
            return "任务等待人工完成验证步骤";
        case task_enums_1.AgentTaskStatus.AWAITING_FINAL_SUBMIT_CONFIRMATION:
            return "任务等待用户进行最终提交确认";
        case task_enums_1.AgentTaskStatus.AWAITING_PAYMENT:
            return "订单已提交，等待支付";
        case task_enums_1.AgentTaskStatus.COMPLETED:
            return "任务已完成";
        case task_enums_1.AgentTaskStatus.FAILED:
            return "任务执行失败";
        case task_enums_1.AgentTaskStatus.CANCELLED:
            return "任务已取消";
        default:
            return `任务状态已切换为 ${status}`;
    }
}
function extractCandidates(result) {
    const data = result.data;
    if (!data || !Array.isArray(data.candidates)) {
        return [];
    }
    return data.candidates;
}
function pickAutoSelectedCandidate(candidates) {
    if (candidates.length === 0) {
        return null;
    }
    return (candidates.find((candidate) => candidate.isRecommended) ??
        [...candidates].sort((left, right) => right.score - left.score)[0] ??
        candidates[0]);
}
function pickAutoSelectedSeatType(intent, candidate) {
    for (const preferredSeatType of intent.seatPreferences) {
        const matchedInventory = candidate.seatInventory.find((inventory) => inventory.seatType === preferredSeatType &&
            inventory.normalizedStatus !== "无" &&
            inventory.normalizedStatus !== "未知");
        if (matchedInventory) {
            return preferredSeatType;
        }
    }
    const firstAvailableSeat = candidate.seatInventory.find((inventory) => inventory.normalizedStatus !== "无" && inventory.normalizedStatus !== "未知");
    return firstAvailableSeat?.seatType ?? intent.seatPreferences[0] ?? "二等座";
}
class InMemoryTaskService {
    repository;
    activeQueuedWorkers = new Set();
    activeSelectWorkers = new Set();
    activeSubmitWorkers = new Set();
    constructor(repository = task_repository_1.inMemoryTaskRepository) {
        this.repository = repository;
    }
    async createTask(input, userId) {
        const now = new Date().toISOString();
        const taskId = `task_${(0, node_crypto_1.randomUUID)()}`;
        const task = {
            id: taskId,
            userId,
            taskType: input.taskType,
            status: task_enums_1.AgentTaskStatus.DRAFT,
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
            id: `evt_${(0, node_crypto_1.randomUUID)()}`,
            taskId: task.id,
            sequenceId: await this.repository.nextSequenceId(task.id),
            type: task_enums_1.TaskEventType.TASK_CREATED,
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
    async getTask(taskId, userId) {
        const task = await this.loadTask(taskId);
        if (task.userId !== userId) {
            throw new app_error_1.AppError(404, error_codes_1.ErrorCode.TASK_NOT_FOUND, "Task not found.");
        }
        return task;
    }
    async getTaskCandidates(taskId, userId) {
        await this.getTask(taskId, userId);
        return this.repository.getCandidates(taskId);
    }
    async getTaskEvents(taskId, userId, afterSequence = 0) {
        await this.getTask(taskId, userId);
        return this.repository.listEvents(taskId, afterSequence);
    }
    async getTaskEventsResponse(taskId, userId, afterSequence = 0) {
        const events = await this.getTaskEvents(taskId, userId, afterSequence);
        const nextSequenceId = await this.getNextSequenceId(taskId, userId);
        return {
            taskId,
            events,
            nextSequenceId
        };
    }
    async getPendingHumanAction(taskId, userId) {
        const task = await this.getTask(taskId, userId);
        switch (task.status) {
            case task_enums_1.AgentTaskStatus.AWAITING_SLOT_CLARIFICATION:
                return {
                    type: task_enums_1.HumanActionType.CLARIFY_SLOTS,
                    checkpointId: `cp_${task.id}_clarify`,
                    prompt: "请补充缺失或存在歧义的订票信息。"
                };
            case task_enums_1.AgentTaskStatus.AWAITING_INTENT_CONFIRMATION:
                return {
                    type: task_enums_1.HumanActionType.CONFIRM_INTENT,
                    checkpointId: `cp_${task.id}_intent_confirm`,
                    prompt: "请确认当前解析出的订票意图。"
                };
            case task_enums_1.AgentTaskStatus.AWAITING_CANDIDATE_CONFIRMATION:
                return {
                    type: task_enums_1.HumanActionType.CONFIRM_CANDIDATE,
                    checkpointId: `cp_${task.id}_candidate_confirm`,
                    prompt: "请选择并确认目标车次。"
                };
            case task_enums_1.AgentTaskStatus.AWAITING_HUMAN_VERIFICATION:
                return {
                    type: task_enums_1.HumanActionType.COMPLETE_QR_LOGIN,
                    checkpointId: getHumanVerificationCheckpointId(task.id),
                    prompt: "请在 12306 页面完成扫码登录或验证码验证。"
                };
            case task_enums_1.AgentTaskStatus.AWAITING_FINAL_SUBMIT_CONFIRMATION:
                return {
                    type: task_enums_1.HumanActionType.CONFIRM_FINAL_SUBMIT,
                    checkpointId: getFinalSubmitCheckpointId(task.id),
                    prompt: "请确认是否提交订单。"
                };
            case task_enums_1.AgentTaskStatus.AWAITING_PAYMENT:
                return {
                    type: task_enums_1.HumanActionType.COMPLETE_PAYMENT,
                    checkpointId: `cp_${task.id}_payment`,
                    prompt: "请在 12306 页面完成支付。"
                };
            default:
                return null;
        }
    }
    async submitClarifications() {
        throw new app_error_1.AppError(501, error_codes_1.ErrorCode.INTERNAL_ERROR, "submitClarifications is not implemented yet.");
    }
    async confirmIntent(taskId, input, userId) {
        const task = await this.getTask(taskId, userId);
        if (input.confirmed) {
            await this.transition(task.id, task_enums_1.AgentTaskStatus.QUEUED, {
                eventMessage: "用户已确认订票意图，任务进入排队状态",
                eventPayload: {
                    confirmed: true
                }
            });
        }
        else {
            await this.transition(task.id, task_enums_1.AgentTaskStatus.CANCELLED, {
                terminalReason: "intent_rejected",
                eventMessage: "用户拒绝当前订票意图，任务已取消",
                eventPayload: {
                    confirmed: false
                }
            });
        }
        return this.getTask(task.id, userId);
    }
    async confirmCandidate(taskId, input, userId, options = {}) {
        const task = await this.getTask(taskId, userId);
        const candidates = await this.repository.getCandidates(taskId);
        const matchedCandidate = candidates.find((candidate) => candidate.trainNo === input.candidateTrainNo);
        const autoConfirmed = options.autoConfirmed ?? false;
        if (!matchedCandidate) {
            throw new app_error_1.AppError(404, error_codes_1.ErrorCode.NO_TRAIN_MATCHED, "Candidate train not found.");
        }
        if (task.status !== task_enums_1.AgentTaskStatus.SEARCH_RESULTS_READY &&
            task.status !== task_enums_1.AgentTaskStatus.AWAITING_CANDIDATE_CONFIRMATION) {
            throw new app_error_1.AppError(409, error_codes_1.ErrorCode.TASK_STATUS_INVALID, `Task is not ready for candidate confirmation: ${task.status}.`);
        }
        if (task.status === task_enums_1.AgentTaskStatus.SEARCH_RESULTS_READY) {
            await this.transition(taskId, task_enums_1.AgentTaskStatus.AWAITING_CANDIDATE_CONFIRMATION, {
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
        const nextIntent = {
            ...latestTask.intent,
            seatPreferences: [
                input.seatType,
                ...latestTask.intent.seatPreferences.filter((seatType) => seatType !== input.seatType)
            ],
            allowWaitlist: input.allowWaitlist
        };
        await this.transition(taskId, task_enums_1.AgentTaskStatus.SELECTING_TRAIN, {
            selectedCandidateId: input.candidateTrainNo,
            intent: nextIntent,
            eventType: task_enums_1.TaskEventType.CANDIDATE_CONFIRMED,
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
    async markHumanStepDone(taskId, input, userId) {
        const task = await this.getTask(taskId, userId);
        const expectedCheckpointId = getHumanVerificationCheckpointId(taskId);
        const allowedActions = new Set([
            task_enums_1.HumanActionType.COMPLETE_QR_LOGIN,
            task_enums_1.HumanActionType.COMPLETE_CAPTCHA,
            task_enums_1.HumanActionType.COMPLETE_SMS_VERIFICATION
        ]);
        if (task.status !== task_enums_1.AgentTaskStatus.AWAITING_HUMAN_VERIFICATION) {
            throw new app_error_1.AppError(409, error_codes_1.ErrorCode.TASK_STATUS_INVALID, `Task is not awaiting human verification: ${task.status}.`);
        }
        if (input.checkpointId !== expectedCheckpointId) {
            throw new app_error_1.AppError(409, error_codes_1.ErrorCode.TASK_STATUS_INVALID, "Checkpoint ID does not match pending human action.", {
                expectedCheckpointId,
                actualCheckpointId: input.checkpointId
            });
        }
        if (!allowedActions.has(input.actionType)) {
            throw new app_error_1.AppError(409, error_codes_1.ErrorCode.TASK_STATUS_INVALID, "Unsupported human action type for this step.", {
                actionType: input.actionType
            });
        }
        browser_service_1.browserService.markHumanVerified(taskId);
        await this.transition(taskId, task_enums_1.AgentTaskStatus.SELECTING_TRAIN, {
            eventType: task_enums_1.TaskEventType.HUMAN_STEP_COMPLETED,
            eventMessage: "人工验证已完成，恢复选座流程",
            eventPayload: {
                checkpointId: input.checkpointId,
                actionType: input.actionType
            }
        });
        this.scheduleSelectTrain(taskId);
        return this.getTask(taskId, userId);
    }
    async confirmSubmit(taskId, input, userId) {
        const task = await this.getTask(taskId, userId);
        const expectedCheckpointId = getFinalSubmitCheckpointId(taskId);
        if (task.status !== task_enums_1.AgentTaskStatus.AWAITING_FINAL_SUBMIT_CONFIRMATION) {
            throw new app_error_1.AppError(409, error_codes_1.ErrorCode.TASK_STATUS_INVALID, `Task is not awaiting final submit confirmation: ${task.status}.`);
        }
        if (input.checkpointId !== expectedCheckpointId) {
            throw new app_error_1.AppError(409, error_codes_1.ErrorCode.TASK_STATUS_INVALID, "Checkpoint ID does not match pending submit step.", {
                expectedCheckpointId,
                actualCheckpointId: input.checkpointId
            });
        }
        if (input.confirmed) {
            await this.transition(taskId, task_enums_1.AgentTaskStatus.SUBMITTING_ORDER, {
                checkpointId: expectedCheckpointId,
                eventType: task_enums_1.TaskEventType.FINAL_SUBMIT_CONFIRMED,
                eventMessage: "用户已确认最终提交订单，开始执行提交流程",
                eventPayload: {
                    checkpointId: input.checkpointId,
                    confirmed: true
                }
            });
        }
        else {
            await this.transition(taskId, task_enums_1.AgentTaskStatus.CANCELLED, {
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
    async completePayment(taskId, input, userId) {
        const task = await this.getTask(taskId, userId);
        const expectedCheckpointId = getPaymentCheckpointId(taskId);
        if (task.status !== task_enums_1.AgentTaskStatus.AWAITING_PAYMENT) {
            throw new app_error_1.AppError(409, error_codes_1.ErrorCode.TASK_STATUS_INVALID, `Task is not awaiting payment: ${task.status}.`);
        }
        if (input.checkpointId !== expectedCheckpointId) {
            throw new app_error_1.AppError(409, error_codes_1.ErrorCode.TASK_STATUS_INVALID, "Checkpoint ID does not match payment step.", {
                expectedCheckpointId,
                actualCheckpointId: input.checkpointId
            });
        }
        await this.transition(taskId, task_enums_1.AgentTaskStatus.COMPLETED, {
            checkpointId: expectedCheckpointId,
            eventMessage: "支付成功，祝您旅途愉快！",
            eventPayload: {
                checkpointId: input.checkpointId,
                actionType: task_enums_1.HumanActionType.COMPLETE_PAYMENT
            },
            requiresAck: false
        });
        return this.getTask(taskId, userId);
    }
    async cancelTask(taskId, input, userId) {
        const task = await this.getTask(taskId, userId);
        if (isTerminalStatus(task.status)) {
            throw new app_error_1.AppError(409, error_codes_1.ErrorCode.TASK_STATUS_INVALID, `Task is already terminal and cannot be cancelled: ${task.status}.`);
        }
        await this.transition(taskId, task_enums_1.AgentTaskStatus.CANCELLED, {
            terminalReason: input.reason ?? "cancelled_by_user",
            eventMessage: "用户已取消当前任务",
            eventPayload: {
                reason: input.reason ?? null
            }
        });
        return this.getTask(taskId, userId);
    }
    async appendEvent(taskId, event) {
        await this.loadTask(taskId);
        await this.repository.appendEvent(event);
    }
    async transition(taskId, nextStatus, meta = {}) {
        const currentTask = await this.loadTask(taskId);
        if (!(0, task_state_machine_1.canTransition)(currentTask.status, nextStatus)) {
            throw new app_error_1.AppError(409, error_codes_1.ErrorCode.TASK_STATUS_INVALID, `Illegal task transition: ${currentTask.status} -> ${nextStatus}.`, {
                from: currentTask.status,
                to: nextStatus
            });
        }
        const now = new Date().toISOString();
        const updatedTask = {
            ...currentTask,
            status: nextStatus,
            intent: meta.intent ?? currentTask.intent,
            context: meta.context ?? currentTask.context,
            selectedCandidateId: meta.selectedCandidateId !== undefined
                ? meta.selectedCandidateId
                : currentTask.selectedCandidateId,
            expiresAt: meta.expiresAt !== undefined ? meta.expiresAt : currentTask.expiresAt,
            terminalReason: meta.terminalReason !== undefined ? meta.terminalReason : currentTask.terminalReason,
            updatedAt: now
        };
        await this.repository.updateTask(updatedTask);
        const event = {
            id: `evt_${(0, node_crypto_1.randomUUID)()}`,
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
        await realtime_service_1.realtimeService.publishTaskEvent(taskId, event);
        if (nextStatus === task_enums_1.AgentTaskStatus.QUEUED) {
            this.scheduleQueuedSearch(taskId);
        }
        if (nextStatus === task_enums_1.AgentTaskStatus.SUBMITTING_ORDER) {
            this.scheduleSubmitOrder(taskId);
        }
    }
    createBrowserProgressReporter(taskId) {
        return async (progress) => {
            const task = await this.repository.getTask(taskId);
            if (!task) {
                return;
            }
            const now = new Date().toISOString();
            const checkpointId = typeof progress.payload?.checkpointId === "string"
                ? progress.payload.checkpointId
                : null;
            const event = {
                id: `evt_${(0, node_crypto_1.randomUUID)()}`,
                taskId,
                sequenceId: await this.repository.nextSequenceId(taskId),
                type: task_enums_1.TaskEventType.BROWSER_PROGRESS,
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
            await realtime_service_1.realtimeService.publishTaskEvent(taskId, event);
        };
    }
    async getNextSequenceId(taskId, userId) {
        await this.getTask(taskId, userId);
        return this.repository.nextSequenceId(taskId);
    }
    async loadTask(taskId) {
        const task = await this.repository.getTask(taskId);
        if (!task) {
            throw new app_error_1.AppError(404, error_codes_1.ErrorCode.TASK_NOT_FOUND, "Task not found.");
        }
        return task;
    }
    scheduleQueuedSearch(taskId) {
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
    scheduleSelectTrain(taskId) {
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
    scheduleSubmitOrder(taskId) {
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
    async runQueuedSearch(taskId) {
        try {
            const currentTask = await this.loadTask(taskId);
            if (currentTask.status !== task_enums_1.AgentTaskStatus.QUEUED) {
                return;
            }
            await this.transition(taskId, task_enums_1.AgentTaskStatus.SEARCHING, {
                eventMessage: "任务已启动浏览器查询流程"
            });
            const latestTask = await this.loadTask(taskId);
            const onProgress = this.createBrowserProgressReporter(taskId);
            const result = await browser_service_1.browserService.execute({
                action: task_enums_1.BrowserActionType.SEARCH_TRAINS,
                taskId,
                payload: {
                    intent: latestTask.intent
                },
                onProgress
            });
            await this.handleBrowserResult(taskId, result);
        }
        catch (error) {
            await this.failTaskIfPossible(taskId, error, "browser_worker_failed", "浏览器查询流程失败。");
        }
    }
    async runSelectTrain(taskId) {
        try {
            const currentTask = await this.loadTask(taskId);
            if (currentTask.status !== task_enums_1.AgentTaskStatus.SELECTING_TRAIN) {
                return;
            }
            if (!currentTask.selectedCandidateId) {
                throw new app_error_1.AppError(409, error_codes_1.ErrorCode.NO_TRAIN_MATCHED, "No candidate selected for train selection.");
            }
            const shouldResumeAfterLogin = browser_service_1.browserService.hasPendingLoginSession(taskId);
            const onProgress = this.createBrowserProgressReporter(taskId);
            const result = await browser_service_1.browserService.execute(shouldResumeAfterLogin
                ? {
                    action: task_enums_1.BrowserActionType.WAIT_LOGIN,
                    taskId,
                    payload: {
                        timeoutMs: 120000
                    },
                    onProgress
                }
                : {
                    action: task_enums_1.BrowserActionType.SELECT_TRAIN,
                    taskId,
                    payload: {
                        intent: currentTask.intent,
                        candidateTrainNo: currentTask.selectedCandidateId,
                        seatType: currentTask.intent.seatPreferences[0] ?? "二等座",
                        passengerNames: currentTask.intent.passengerNames
                    },
                    onProgress
                });
            await this.handleBrowserResult(taskId, result);
        }
        catch (error) {
            await this.failTaskIfPossible(taskId, error, "browser_select_failed", "Browser select flow failed.");
        }
    }
    async runSubmitOrder(taskId) {
        try {
            const currentTask = await this.loadTask(taskId);
            if (currentTask.status !== task_enums_1.AgentTaskStatus.SUBMITTING_ORDER) {
                return;
            }
            const onProgress = this.createBrowserProgressReporter(taskId);
            const result = await browser_service_1.browserService.execute({
                action: task_enums_1.BrowserActionType.SUBMIT_ORDER,
                taskId,
                payload: {},
                onProgress
            });
            await this.handleBrowserResult(taskId, result);
        }
        catch (error) {
            await this.failTaskIfPossible(taskId, error, "browser_submit_failed", "Browser submit flow failed.");
        }
    }
    async handleBrowserResult(taskId, result) {
        if (result.ok && result.state === "search_results_ready") {
            const candidates = extractCandidates(result);
            await this.repository.saveCandidates(taskId, candidates);
            await this.transition(taskId, task_enums_1.AgentTaskStatus.SEARCH_RESULTS_READY, {
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
            await this.confirmCandidate(taskId, {
                idempotencyKey: `auto_confirm_candidate_${(0, node_crypto_1.randomUUID)()}`,
                candidateTrainNo: autoSelectedCandidate.trainNo,
                seatType: pickAutoSelectedSeatType(currentTask.intent, autoSelectedCandidate),
                allowWaitlist: currentTask.intent.allowWaitlist
            }, currentTask.userId, {
                autoConfirmed: true
            });
            return;
        }
        if (result.state === "login_required") {
            await this.transition(taskId, task_enums_1.AgentTaskStatus.LOGIN_REQUIRED, {
                eventMessage: "浏览器流程检测到需要登录",
                eventPayload: result.data,
                eventType: task_enums_1.TaskEventType.LOGIN_REQUIRED
            });
            await this.transition(taskId, task_enums_1.AgentTaskStatus.AWAITING_HUMAN_VERIFICATION, {
                checkpointId: getHumanVerificationCheckpointId(taskId),
                eventType: task_enums_1.TaskEventType.HUMAN_VERIFICATION_REQUIRED,
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
            await this.transition(taskId, task_enums_1.AgentTaskStatus.AWAITING_HUMAN_VERIFICATION, {
                checkpointId: getHumanVerificationCheckpointId(taskId),
                eventType: task_enums_1.TaskEventType.HUMAN_VERIFICATION_REQUIRED,
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
            await this.transition(taskId, task_enums_1.AgentTaskStatus.AWAITING_FINAL_SUBMIT_CONFIRMATION, {
                checkpointId: getFinalSubmitCheckpointId(taskId),
                eventType: task_enums_1.TaskEventType.HUMAN_STEP_COMPLETED,
                eventMessage: "选座成功，系统将自动提交订单",
                eventPayload: {
                    ...(result.data ?? {}),
                    checkpointId: getFinalSubmitCheckpointId(taskId),
                    autoSubmit: true
                },
                requiresAck: false
            });
            await this.transition(taskId, task_enums_1.AgentTaskStatus.SUBMITTING_ORDER, {
                checkpointId: getFinalSubmitCheckpointId(taskId),
                eventType: task_enums_1.TaskEventType.FINAL_SUBMIT_CONFIRMED,
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
            await this.transition(taskId, task_enums_1.AgentTaskStatus.AWAITING_PAYMENT, {
                checkpointId: getPaymentCheckpointId(taskId),
                eventType: task_enums_1.TaskEventType.ORDER_SUBMITTED,
                eventMessage: "订单已提交，等待用户完成支付",
                eventPayload: {
                    ...(result.data ?? {}),
                    checkpointId: getPaymentCheckpointId(taskId)
                },
                requiresAck: true
            });
            return;
        }
        throw new app_error_1.AppError(500, result.errorCode ?? error_codes_1.ErrorCode.INTERNAL_ERROR, result.errorMessage ?? "Browser action failed.");
    }
    async failTaskIfPossible(taskId, error, terminalReason, fallbackMessage) {
        const latestTask = await this.repository.getTask(taskId);
        if (!latestTask ||
            [task_enums_1.AgentTaskStatus.FAILED, task_enums_1.AgentTaskStatus.CANCELLED, task_enums_1.AgentTaskStatus.EXPIRED].includes(latestTask.status)) {
            return;
        }
        const message = error instanceof Error ? error.message : fallbackMessage;
        const code = error instanceof app_error_1.AppError ? error.code : error_codes_1.ErrorCode.INTERNAL_ERROR;
        await this.transition(taskId, task_enums_1.AgentTaskStatus.FAILED, {
            terminalReason,
            eventMessage: message,
            eventPayload: {
                code
            }
        });
    }
}
exports.InMemoryTaskService = InMemoryTaskService;
exports.taskService = new InMemoryTaskService();

"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.canTransition = canTransition;
const task_enums_1 = require("./task.enums");
function canTransition(from, to) {
    const allowed = {
        [task_enums_1.AgentTaskStatus.DRAFT]: [task_enums_1.AgentTaskStatus.PARSING_INTENT, task_enums_1.AgentTaskStatus.CANCELLED],
        [task_enums_1.AgentTaskStatus.PARSING_INTENT]: [
            task_enums_1.AgentTaskStatus.AWAITING_SLOT_CLARIFICATION,
            task_enums_1.AgentTaskStatus.AWAITING_INTENT_CONFIRMATION,
            task_enums_1.AgentTaskStatus.FAILED,
            task_enums_1.AgentTaskStatus.CANCELLED
        ],
        [task_enums_1.AgentTaskStatus.AWAITING_SLOT_CLARIFICATION]: [
            task_enums_1.AgentTaskStatus.PARSING_INTENT,
            task_enums_1.AgentTaskStatus.CANCELLED,
            task_enums_1.AgentTaskStatus.EXPIRED
        ],
        [task_enums_1.AgentTaskStatus.AWAITING_INTENT_CONFIRMATION]: [
            task_enums_1.AgentTaskStatus.QUEUED,
            task_enums_1.AgentTaskStatus.CANCELLED,
            task_enums_1.AgentTaskStatus.EXPIRED
        ],
        [task_enums_1.AgentTaskStatus.QUEUED]: [
            task_enums_1.AgentTaskStatus.SEARCHING,
            task_enums_1.AgentTaskStatus.CANCELLED,
            task_enums_1.AgentTaskStatus.FAILED
        ],
        [task_enums_1.AgentTaskStatus.SEARCHING]: [
            task_enums_1.AgentTaskStatus.SEARCH_RESULTS_READY,
            task_enums_1.AgentTaskStatus.FAILED,
            task_enums_1.AgentTaskStatus.CANCELLED
        ],
        [task_enums_1.AgentTaskStatus.SEARCH_RESULTS_READY]: [
            task_enums_1.AgentTaskStatus.AWAITING_CANDIDATE_CONFIRMATION,
            task_enums_1.AgentTaskStatus.FAILED,
            task_enums_1.AgentTaskStatus.CANCELLED
        ],
        [task_enums_1.AgentTaskStatus.AWAITING_CANDIDATE_CONFIRMATION]: [
            task_enums_1.AgentTaskStatus.SELECTING_TRAIN,
            task_enums_1.AgentTaskStatus.CANCELLED,
            task_enums_1.AgentTaskStatus.EXPIRED
        ],
        [task_enums_1.AgentTaskStatus.LOGIN_REQUIRED]: [
            task_enums_1.AgentTaskStatus.AWAITING_HUMAN_VERIFICATION,
            task_enums_1.AgentTaskStatus.FAILED,
            task_enums_1.AgentTaskStatus.CANCELLED
        ],
        [task_enums_1.AgentTaskStatus.AWAITING_HUMAN_VERIFICATION]: [
            task_enums_1.AgentTaskStatus.SELECTING_TRAIN,
            task_enums_1.AgentTaskStatus.CANCELLED,
            task_enums_1.AgentTaskStatus.EXPIRED
        ],
        [task_enums_1.AgentTaskStatus.SELECTING_TRAIN]: [
            task_enums_1.AgentTaskStatus.LOGIN_REQUIRED,
            task_enums_1.AgentTaskStatus.AWAITING_FINAL_SUBMIT_CONFIRMATION,
            task_enums_1.AgentTaskStatus.FAILED,
            task_enums_1.AgentTaskStatus.CANCELLED
        ],
        [task_enums_1.AgentTaskStatus.AWAITING_FINAL_SUBMIT_CONFIRMATION]: [
            task_enums_1.AgentTaskStatus.SUBMITTING_ORDER,
            task_enums_1.AgentTaskStatus.CANCELLED,
            task_enums_1.AgentTaskStatus.EXPIRED
        ],
        [task_enums_1.AgentTaskStatus.SUBMITTING_ORDER]: [
            task_enums_1.AgentTaskStatus.AWAITING_PAYMENT,
            task_enums_1.AgentTaskStatus.FAILED,
            task_enums_1.AgentTaskStatus.CANCELLED
        ],
        [task_enums_1.AgentTaskStatus.AWAITING_PAYMENT]: [
            task_enums_1.AgentTaskStatus.COMPLETED,
            task_enums_1.AgentTaskStatus.CANCELLED,
            task_enums_1.AgentTaskStatus.EXPIRED
        ],
        [task_enums_1.AgentTaskStatus.COMPLETED]: [],
        [task_enums_1.AgentTaskStatus.FAILED]: [],
        [task_enums_1.AgentTaskStatus.CANCELLED]: [],
        [task_enums_1.AgentTaskStatus.EXPIRED]: []
    };
    return allowed[from]?.includes(to) ?? false;
}

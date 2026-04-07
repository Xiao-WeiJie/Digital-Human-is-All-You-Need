import { AgentTaskStatus } from "./task.enums";

export function canTransition(from: AgentTaskStatus, to: AgentTaskStatus): boolean {
  const allowed: Record<AgentTaskStatus, AgentTaskStatus[]> = {
    [AgentTaskStatus.DRAFT]: [AgentTaskStatus.PARSING_INTENT, AgentTaskStatus.CANCELLED],
    [AgentTaskStatus.PARSING_INTENT]: [
      AgentTaskStatus.AWAITING_SLOT_CLARIFICATION,
      AgentTaskStatus.AWAITING_INTENT_CONFIRMATION,
      AgentTaskStatus.FAILED,
      AgentTaskStatus.CANCELLED
    ],
    [AgentTaskStatus.AWAITING_SLOT_CLARIFICATION]: [
      AgentTaskStatus.PARSING_INTENT,
      AgentTaskStatus.CANCELLED,
      AgentTaskStatus.EXPIRED
    ],
    [AgentTaskStatus.AWAITING_INTENT_CONFIRMATION]: [
      AgentTaskStatus.QUEUED,
      AgentTaskStatus.CANCELLED,
      AgentTaskStatus.EXPIRED
    ],
    [AgentTaskStatus.QUEUED]: [
      AgentTaskStatus.SEARCHING,
      AgentTaskStatus.CANCELLED,
      AgentTaskStatus.FAILED
    ],
    [AgentTaskStatus.SEARCHING]: [
      AgentTaskStatus.SEARCH_RESULTS_READY,
      AgentTaskStatus.FAILED,
      AgentTaskStatus.CANCELLED
    ],
    [AgentTaskStatus.SEARCH_RESULTS_READY]: [
      AgentTaskStatus.AWAITING_CANDIDATE_CONFIRMATION,
      AgentTaskStatus.FAILED,
      AgentTaskStatus.CANCELLED
    ],
    [AgentTaskStatus.AWAITING_CANDIDATE_CONFIRMATION]: [
      AgentTaskStatus.SELECTING_TRAIN,
      AgentTaskStatus.CANCELLED,
      AgentTaskStatus.EXPIRED
    ],
    [AgentTaskStatus.LOGIN_REQUIRED]: [
      AgentTaskStatus.AWAITING_HUMAN_VERIFICATION,
      AgentTaskStatus.FAILED,
      AgentTaskStatus.CANCELLED
    ],
    [AgentTaskStatus.AWAITING_HUMAN_VERIFICATION]: [
      AgentTaskStatus.SELECTING_TRAIN,
      AgentTaskStatus.CANCELLED,
      AgentTaskStatus.EXPIRED
    ],
    [AgentTaskStatus.SELECTING_TRAIN]: [
      AgentTaskStatus.LOGIN_REQUIRED,
      AgentTaskStatus.AWAITING_FINAL_SUBMIT_CONFIRMATION,
      AgentTaskStatus.FAILED,
      AgentTaskStatus.CANCELLED
    ],
    [AgentTaskStatus.AWAITING_FINAL_SUBMIT_CONFIRMATION]: [
      AgentTaskStatus.SUBMITTING_ORDER,
      AgentTaskStatus.CANCELLED,
      AgentTaskStatus.EXPIRED
    ],
    [AgentTaskStatus.SUBMITTING_ORDER]: [
      AgentTaskStatus.AWAITING_PAYMENT,
      AgentTaskStatus.FAILED,
      AgentTaskStatus.CANCELLED
    ],
    [AgentTaskStatus.AWAITING_PAYMENT]: [
      AgentTaskStatus.COMPLETED,
      AgentTaskStatus.CANCELLED,
      AgentTaskStatus.EXPIRED
    ],
    [AgentTaskStatus.COMPLETED]: [],
    [AgentTaskStatus.FAILED]: [],
    [AgentTaskStatus.CANCELLED]: [],
    [AgentTaskStatus.EXPIRED]: []
  };

  return allowed[from]?.includes(to) ?? false;
}

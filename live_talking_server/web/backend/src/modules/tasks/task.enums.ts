export enum AgentTaskStatus {
  DRAFT = "draft",
  PARSING_INTENT = "parsing_intent",
  AWAITING_SLOT_CLARIFICATION = "awaiting_slot_clarification",
  AWAITING_INTENT_CONFIRMATION = "awaiting_intent_confirmation",
  QUEUED = "queued",
  SEARCHING = "searching",
  SEARCH_RESULTS_READY = "search_results_ready",
  AWAITING_CANDIDATE_CONFIRMATION = "awaiting_candidate_confirmation",
  LOGIN_REQUIRED = "login_required",
  AWAITING_HUMAN_VERIFICATION = "awaiting_human_verification",
  SELECTING_TRAIN = "selecting_train",
  AWAITING_FINAL_SUBMIT_CONFIRMATION = "awaiting_final_submit_confirmation",
  SUBMITTING_ORDER = "submitting_order",
  AWAITING_PAYMENT = "awaiting_payment",
  COMPLETED = "completed",
  FAILED = "failed",
  CANCELLED = "cancelled",
  EXPIRED = "expired"
}

export enum CheckpointStatus {
  PENDING = "pending",
  RUNNING = "running",
  SUCCEEDED = "succeeded",
  FAILED = "failed",
  SKIPPED = "skipped",
  WAITING_HUMAN = "waiting_human"
}

export enum TaskEventType {
  TASK_CREATED = "task_created",
  INTENT_PARSED = "intent_parsed",
  SLOT_CLARIFICATION_REQUESTED = "slot_clarification_requested",
  SLOT_CLARIFICATION_RECEIVED = "slot_clarification_received",
  INTENT_CONFIRMED = "intent_confirmed",
  SEARCH_STARTED = "search_started",
  BROWSER_PROGRESS = "browser_progress",
  SEARCH_RESULTS_READY = "search_results_ready",
  CANDIDATE_CONFIRMED = "candidate_confirmed",
  LOGIN_REQUIRED = "login_required",
  HUMAN_VERIFICATION_REQUIRED = "human_verification_required",
  HUMAN_STEP_COMPLETED = "human_step_completed",
  FINAL_SUBMIT_CONFIRMED = "final_submit_confirmed",
  ORDER_SUBMITTED = "order_submitted",
  PAYMENT_PENDING = "payment_pending",
  TASK_COMPLETED = "task_completed",
  TASK_FAILED = "task_failed",
  TASK_CANCELLED = "task_cancelled"
}

export enum HumanActionType {
  CONFIRM_INTENT = "confirm_intent",
  CLARIFY_SLOTS = "clarify_slots",
  CONFIRM_CANDIDATE = "confirm_candidate",
  COMPLETE_QR_LOGIN = "complete_qr_login",
  COMPLETE_CAPTCHA = "complete_captcha",
  COMPLETE_SMS_VERIFICATION = "complete_sms_verification",
  CONFIRM_FINAL_SUBMIT = "confirm_final_submit",
  COMPLETE_PAYMENT = "complete_payment",
  CANCEL_TASK = "cancel_task"
}

export enum BrowserActionType {
  SEARCH_TRAINS = "search_trains",
  WAIT_LOGIN = "wait_login",
  SELECT_TRAIN = "select_train",
  SELECT_PASSENGERS = "select_passengers",
  CONFIRM_ORDER_PAGE = "confirm_order_page",
  SUBMIT_ORDER = "submit_order",
  WAIT_PAYMENT_PAGE = "wait_payment_page"
}

export enum CheckpointName {
  OPEN_LEFT_TICKET_PAGE = "open_left_ticket_page",
  FILL_ROUTE_FORM = "fill_route_form",
  SUBMIT_SEARCH = "submit_search",
  PARSE_SEARCH_RESULTS = "parse_search_results",
  CHECK_LOGIN_STATUS = "check_login_status",
  WAIT_QR_LOGIN = "wait_qr_login",
  SELECT_TRAIN_ROW = "select_train_row",
  SELECT_PASSENGERS = "select_passengers",
  VERIFY_ORDER_PAGE = "verify_order_page",
  FINAL_SUBMIT_WAIT = "final_submit_wait",
  SUBMIT_ORDER = "submit_order",
  WAIT_PAYMENT_PAGE = "wait_payment_page"
}

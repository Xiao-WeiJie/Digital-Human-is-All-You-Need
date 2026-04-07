import { type ErrorCode } from "../../shared/errors/error-codes";
import { type ID } from "../../shared/types/common";
import {
  type BookingIntent,
  type SeatType,
  type TrainCandidate
} from "../booking/booking.types";
import { BrowserActionType } from "../tasks/task.enums";

export type BrowserProgressLevel = "TRACE" | "URL" | "DOM" | "DATA" | "HIT" | "OK" | "WARN";

export type BrowserProgressEvent = {
  level: BrowserProgressLevel;
  message: string;
  payload?: Record<string, unknown>;
};

export type BrowserProgressReporter = (event: BrowserProgressEvent) => Promise<void> | void;

type BrowserActionBase<TAction extends BrowserActionType, TPayload> = {
  action: TAction;
  taskId: ID;
  payload: TPayload;
  onProgress?: BrowserProgressReporter;
};

export type BrowserSearchTrainsResultData = {
  candidates: TrainCandidate[];
  openedUrl: string;
  navigationSucceeded: boolean;
};

export type BrowserLoginRequiredResultData = {
  timeoutMs: number;
  openedUrl?: string;
};

export type BrowserActionRequest =
  | BrowserActionBase<
      BrowserActionType.SEARCH_TRAINS,
      {
        intent: BookingIntent;
      }
    >
  | BrowserActionBase<
      BrowserActionType.WAIT_LOGIN,
      {
        timeoutMs: number;
      }
    >
  | BrowserActionBase<
      BrowserActionType.SELECT_TRAIN,
      {
        intent: BookingIntent;
        candidateTrainNo: string;
        seatType: SeatType;
        passengerNames: string[];
      }
    >
  | BrowserActionBase<BrowserActionType.CONFIRM_ORDER_PAGE, Record<string, never>>
  | BrowserActionBase<BrowserActionType.SUBMIT_ORDER, Record<string, never>>;

export type BrowserActionResult = {
  ok: boolean;
  state:
    | "search_results_ready"
    | "login_required"
    | "waiting_human"
    | "train_selected"
    | "order_page_ready"
    | "order_submitted"
    | "failed";
  screenshotUrl?: string;
  traceId?: string;
  errorCode?: ErrorCode;
  errorMessage?: string;
  data?: Record<string, unknown>;
};

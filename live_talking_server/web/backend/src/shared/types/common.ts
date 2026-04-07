import { type ErrorCode } from "../errors/error-codes";

export type ID = string;

export type ISODate = string;
export type ISODateTime = string;
export type TimeHHMM = string;

export type Nullable<T> = T | null;

export type ApiResponse<T> = {
  success: boolean;
  requestId: string;
  data?: T;
  error?: {
    code: ErrorCode | string;
    message: string;
    details?: unknown;
  };
};

import { type ID, type ISODate, type Nullable, type TimeHHMM } from "../../shared/types/common";

export type TrainTypeCode = "G" | "D" | "C" | "Z" | "T" | "K" | "OTHER";

export type SeatType =
  | "商务座"
  | "特等座"
  | "一等座"
  | "二等座"
  | "高级软卧"
  | "软卧"
  | "硬卧"
  | "软座"
  | "硬座"
  | "无座";

export type BookingIntent = {
  rawText: string;
  travelDate: Nullable<ISODate>;
  origin: Nullable<string>;
  destination: Nullable<string>;
  departureTimeLowerBound: Nullable<TimeHHMM>;
  departureTimeUpperBound: Nullable<TimeHHMM>;
  trainTypes: TrainTypeCode[];
  seatPreferences: SeatType[];
  passengerNames: string[];
  allowWaitlist: boolean;
  allowFallbackTime: boolean;
  allowFallbackSeat: boolean;
  queryOnly: boolean;
};

export type BookingSlotName =
  | "travelDate"
  | "origin"
  | "destination"
  | "departureTimeLowerBound"
  | "departureTimeUpperBound"
  | "seatPreferences"
  | "passengerNames";

export type SlotConfidence = {
  slot: BookingSlotName;
  confidence: number;
  source: "rule" | "llm" | "user_confirmed";
};

export type ConversationContext = {
  taskId: ID;
  turns: number;
  missingSlots: BookingSlotName[];
  ambiguousSlots: BookingSlotName[];
  slotConfidence: SlotConfidence[];
  lastClarificationQuestion: Nullable<string>;
};

export type SeatInventoryStatus = "有" | "无" | "候补" | "数字" | "未知";

export type SeatInventoryItem = {
  seatType: SeatType;
  rawValue: string;
  normalizedStatus: SeatInventoryStatus;
};

export type TrainCandidate = {
  trainNo: string;
  originStation: string;
  destinationStation: string;
  departureTime: TimeHHMM;
  arrivalTime: TimeHHMM;
  durationText: string;
  trainType: TrainTypeCode;
  seatInventory: SeatInventoryItem[];
  score: number;
  isRecommended: boolean;
};

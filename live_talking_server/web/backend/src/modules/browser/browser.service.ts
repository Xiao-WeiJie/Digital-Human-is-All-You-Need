import { chromium, type Browser, type BrowserContext, type Locator, type Page } from "playwright";

import { ErrorCode } from "../../shared/errors/error-codes";
import { type ID } from "../../shared/types/common";
import {
  type BookingIntent,
  type SeatInventoryStatus,
  type SeatType,
  type TrainCandidate,
  type TrainTypeCode
} from "../booking/booking.types";
import { BrowserActionType } from "../tasks/task.enums";
import {
  type BrowserActionRequest,
  type BrowserActionResult,
  type BrowserProgressLevel,
  type BrowserProgressReporter,
  type BrowserSearchTrainsResultData
} from "./browser.types";

const LEFT_TICKET_URL = "https://kyfw.12306.cn/otn/leftTicket/init";
const SEARCH_VISUAL_WAIT_MS = 3000;
const DEFAULT_VISUAL_WAIT_MS = 2000;
const CITY_INPUT_DELAY_MS = 150;
const RESULT_WAIT_TIMEOUT_MS = 30000;
const HIGH_SPEED_STATION_SUFFIXES = ["北", "东", "南", "西", "龙门"] as const;
const HIGH_SPEED_STATION_OVERRIDES: Record<string, string[]> = {
  资阳: ["资阳北"],
  洛阳: ["洛阳龙门"]
};

const FROM_STATION_INPUT_SELECTOR = "#fromStationText";
const TO_STATION_INPUT_SELECTOR = "#toStationText";
const FROM_STATION_CODE_SELECTOR = "#fromStation";
const TO_STATION_CODE_SELECTOR = "#toStation";
const TRAIN_DATE_SELECTOR = "#train_date";
const QUERY_BUTTON_SELECTOR = "#query_ticket";
const RESULT_TABLE_SELECTOR = "#queryLeftTable";
const RESULT_ROW_SELECTOR = "#queryLeftTable tr[id^='ticket_']";
const CITY_PANEL_SELECTORS = ["#panel_cities", ".station_select", ".city-select"];
const LOGIN_MODAL_SELECTORS = [
  "#login.modal-login",
  "#login",
  ".modal-login",
  ".login-box",
  ".login-hd",
  ".login-account"
];
const PREORDER_TEXT = "\u9884\u8ba2";

type BrowserPageSession = {
  browser: Browser;
  context: BrowserContext;
  page: Page;
};

type ParsedTrainCandidate = {
  trainNo: string;
  originStation: string;
  destinationStation: string;
  departureTime: string;
  arrivalTime: string;
  durationText: string;
  firstClassStatus: string;
  secondClassStatus: string;
};

type ResolvedStation = {
  stationName: string;
  stationCode: string;
  matchedTarget: string;
};

const NOOP_PROGRESS_REPORTER: BrowserProgressReporter = () => undefined;

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }

  return String(error);
}

function normalizeText(value: string | null | undefined): string {
  return (value ?? "").replace(/\s+/g, " ").trim();
}

function normalizeCompactText(value: string | null | undefined): string {
  return normalizeText(value).replace(/\s+/g, "");
}

function buildStationSearchTargets(city: string, preferHighSpeedStation: boolean): string[] {
  const normalizedCity = normalizeText(city);
  if (!normalizedCity) {
    return [];
  }

  const targets = new Set<string>([normalizedCity]);

  if (preferHighSpeedStation) {
    const overrides = HIGH_SPEED_STATION_OVERRIDES[normalizedCity] ?? [];
    for (const override of overrides) {
      targets.add(override);
    }

    for (const suffix of HIGH_SPEED_STATION_SUFFIXES) {
      targets.add(`${normalizedCity}${suffix}`);
    }
  }

  return Array.from(targets);
}

async function emitBrowserProgress(
  reporter: BrowserProgressReporter | undefined,
  level: BrowserProgressLevel,
  message: string,
  payload: Record<string, unknown> = {}
): Promise<void> {
  const safeReporter = reporter ?? NOOP_PROGRESS_REPORTER;

  try {
    await safeReporter({
      level,
      message,
      payload
    });
  } catch {
    // Ignore progress-reporting errors to avoid interrupting the browser flow.
  }
}

function extractLeadingChineseName(value: string): string {
  const matched = normalizeText(value).match(/^[\u3400-\u9FFF路]+/u);
  return matched?.[0] ?? "";
}

function scoreCityOption(optionText: string, targetCity: string): number {
  const normalizedTarget = normalizeCompactText(targetCity).toLowerCase();
  const normalizedText = normalizeCompactText(optionText).toLowerCase();
  const chineseName = normalizeCompactText(extractLeadingChineseName(optionText));

  if (!normalizedText) {
    return -1;
  }

  if (
    normalizedText !== normalizedTarget &&
    chineseName !== normalizedTarget &&
    !normalizedText.includes(normalizedTarget) &&
    !chineseName.includes(normalizedTarget)
  ) {
    return -1;
  }

  if (chineseName === normalizedTarget) {
    return 400;
  }

  if (normalizedText === normalizedTarget) {
    return 320;
  }

  if (chineseName.startsWith(normalizedTarget)) {
    return 260 - chineseName.length;
  }

  if (normalizedText.startsWith(normalizedTarget)) {
    return 220 - normalizedText.length;
  }

  if (chineseName.includes(normalizedTarget)) {
    return 180 - chineseName.length;
  }

  return 120 - normalizedText.length;
}

function inferTrainType(trainNo: string): TrainTypeCode {
  const leading = normalizeText(trainNo).charAt(0).toUpperCase();

  switch (leading) {
    case "G":
    case "D":
    case "C":
    case "Z":
    case "T":
    case "K":
      return leading;
    default:
      return "OTHER";
  }
}

function normalizeSeatStatus(rawValue: string): SeatInventoryStatus {
  const normalized = normalizeCompactText(rawValue);

  if (!normalized || normalized === "--" || normalized === "*" || normalized === "暂无") {
    return "未知" as SeatInventoryStatus;
  }

  if (normalized.includes("候补")) {
    return "候补" as SeatInventoryStatus;
  }

  if (normalized === "有") {
    return "有" as SeatInventoryStatus;
  }

  if (normalized === "无") {
    return "无" as SeatInventoryStatus;
  }

  if (/^\d+$/.test(normalized)) {
    return "数字" as SeatInventoryStatus;
  }

  return "未知" as SeatInventoryStatus;
}

function toSeatType(value: string): SeatType {
  return value as SeatType;
}

function buildSeatInventory(seatType: string, rawValue: string) {
  return {
    seatType: toSeatType(seatType),
    rawValue: normalizeText(rawValue) || "--",
    normalizedStatus: normalizeSeatStatus(rawValue)
  };
}

function calculateCandidateScore(candidate: ParsedTrainCandidate, index: number): number {
  let score = 80 - index * 6;

  const secondClassStatus = normalizeSeatStatus(candidate.secondClassStatus);
  const firstClassStatus = normalizeSeatStatus(candidate.firstClassStatus);

  if (secondClassStatus === ("有" as SeatInventoryStatus)) {
    score += 16;
  } else if (secondClassStatus === ("数字" as SeatInventoryStatus)) {
    score += 12;
  } else if (secondClassStatus === ("候补" as SeatInventoryStatus)) {
    score += 8;
  }

  if (firstClassStatus === ("有" as SeatInventoryStatus)) {
    score += 8;
  } else if (firstClassStatus === ("数字" as SeatInventoryStatus)) {
    score += 5;
  } else if (firstClassStatus === ("候补" as SeatInventoryStatus)) {
    score += 3;
  }

  return Math.max(1, Math.min(score, 99));
}

async function createBrowserPage(): Promise<BrowserPageSession> {
  const browser = await chromium.launch({
    headless: false
  });

  const context = await browser.newContext({
    ignoreHTTPSErrors: true
  });

  const page = await context.newPage();

  return { browser, context, page };
}

async function closeBrowserPage(session: BrowserPageSession): Promise<void> {
  await session.context.close();
  await session.browser.close();
}

async function gotoLeftTicketPage(page: Page): Promise<boolean> {
  try {
    await page.goto(LEFT_TICKET_URL, {
      waitUntil: "domcontentloaded",
      timeout: 45000
    });

    await page.waitForLoadState("load", {
      timeout: 15000
    }).catch(() => undefined);

    return true;
  } catch {
    return false;
  }
}

async function waitForVisibleSelector(
  page: Page,
  selectors: string[],
  timeoutMs = 5000
): Promise<string | null> {
  for (const selector of selectors) {
    try {
      await page.locator(selector).first().waitFor({
        state: "visible",
        timeout: timeoutMs
      });
      return selector;
    } catch {
      continue;
    }
  }

  return null;
}

async function clearStationInput(page: Page, locator: Locator): Promise<void> {
  await locator.click({
    timeout: 5000
  });

  await page.keyboard.press("Control+A").catch(() => undefined);
  await page.keyboard.press("Backspace").catch(() => undefined);

  await locator
    .evaluate((element) => {
      const input = element as {
        value?: string;
        dispatchEvent(event: Event): boolean;
      };

      if (typeof input.value === "string") {
        input.value = "";
        input.dispatchEvent(new Event("input", { bubbles: true }));
        input.dispatchEvent(new Event("change", { bubbles: true }));
      }
    })
    .catch(() => undefined);
}

async function clickCitySuggestion(page: Page, targetCities: string[]): Promise<boolean> {
  const panelSelector = await waitForVisibleSelector(page, CITY_PANEL_SELECTORS);
  if (!panelSelector) {
    return false;
  }

  const optionSelector = `${panelSelector} div.cityline, ${panelSelector} div.citylineover, ${panelSelector} li, ${panelSelector} a`;
  const startedAt = Date.now();

  while (Date.now() - startedAt < 5000) {
    const options = page.locator(optionSelector);
    const optionCount = Math.min(await options.count(), 80);

    let bestIndex: number | null = null;
    let bestScore = -1;

    for (let index = 0; index < optionCount; index += 1) {
      const option = options.nth(index);
      const isVisible = await option.isVisible().catch(() => false);

      if (!isVisible) {
        continue;
      }

      const optionText = normalizeText(await option.textContent().catch(() => null));
      const score = Math.max(...targetCities.map((targetCity) => scoreCityOption(optionText, targetCity)));

      if (score > bestScore) {
        bestScore = score;
        bestIndex = index;
      }
    }

    if (bestIndex !== null && bestScore >= 0) {
      await options.nth(bestIndex).click({
        timeout: 5000
      });
      return true;
    }

    await page.waitForTimeout(200);
  }

  return false;
}

async function resolveStation(page: Page, targetCities: string[]): Promise<ResolvedStation | null> {
  return page.evaluate((cities) => {
    const stationNames = (globalThis as { station_names?: string }).station_names;
    if (typeof stationNames !== "string" || !stationNames) {
      return null;
    }

    const normalizedTargets = cities.map((city) => city.replace(/\s+/g, "").toLowerCase());
    const entries = stationNames.split("@").filter(Boolean);
    const exactMatches: Array<{ stationName: string; stationCode: string; matchedTarget: string }> = [];
    const fuzzyMatches: Array<{
      stationName: string;
      stationCode: string;
      matchedTarget: string;
      score: number;
    }> = [];

    for (const entry of entries) {
      const fields = entry.split("|");
      const stationName = (fields[1] ?? "").replace(/\s+/g, "");
      const stationCode = (fields[2] ?? "").trim();
      const pinyin = (fields[3] ?? "").replace(/\s+/g, "").toLowerCase();
      const shortPinyin = (fields[4] ?? "").replace(/\s+/g, "").toLowerCase();

      if (!stationName || !stationCode) {
        continue;
      }

      for (const normalizedTarget of normalizedTargets) {
        if (
          stationName.toLowerCase() === normalizedTarget ||
          pinyin === normalizedTarget ||
          shortPinyin === normalizedTarget
        ) {
          exactMatches.push({
            stationName,
            stationCode,
            matchedTarget: normalizedTarget
          });
          continue;
        }

        if (stationName.toLowerCase().startsWith(normalizedTarget)) {
          const suffixLength = stationName.length - normalizedTarget.length;
          fuzzyMatches.push({
            stationName,
            stationCode,
            matchedTarget: normalizedTarget,
            score: 500 - suffixLength
          });
          continue;
        }

        if (stationName.toLowerCase().includes(normalizedTarget)) {
          fuzzyMatches.push({
            stationName,
            stationCode,
            matchedTarget: normalizedTarget,
            score: 300 - stationName.length
          });
        }
      }
    }

    if (exactMatches.length > 0) {
      const bestExact = exactMatches.sort((left, right) => left.stationName.length - right.stationName.length)[0];
      return {
        stationName: bestExact.stationName,
        stationCode: bestExact.stationCode,
        matchedTarget: bestExact.matchedTarget
      };
    }

    if (fuzzyMatches.length > 0) {
      const bestFuzzy = fuzzyMatches.sort((left, right) => right.score - left.score)[0];
      return {
        stationName: bestFuzzy.stationName,
        stationCode: bestFuzzy.stationCode,
        matchedTarget: bestFuzzy.matchedTarget
      };
    }

    return null;
  }, targetCities);
}

async function applyStationFallback(
  page: Page,
  inputSelector: string,
  hiddenCodeSelector: string,
  targetCities: string[]
): Promise<boolean> {
  const resolvedStation = await resolveStation(page, targetCities);

  if (!resolvedStation) {
    return false;
  }

  await page.evaluate(
    ({ nextCity, nextCode, visibleSelector, hiddenSelector }) => {
      const doc = (globalThis as unknown as {
        document?: {
          querySelector(selector: string): unknown;
        };
      }).document;

      const visibleInput = doc?.querySelector(visibleSelector) as
        | {
            value?: string;
            dispatchEvent(event: Event): boolean;
          }
        | null;
      const hiddenInput = doc?.querySelector(hiddenSelector) as
        | {
            value?: string;
            dispatchEvent(event: Event): boolean;
          }
        | null;

      if (visibleInput && typeof visibleInput.value === "string") {
        visibleInput.value = nextCity;
        visibleInput.dispatchEvent(new Event("input", { bubbles: true }));
        visibleInput.dispatchEvent(new Event("change", { bubbles: true }));
        visibleInput.dispatchEvent(new Event("blur", { bubbles: true }));
      }

      if (hiddenInput && typeof hiddenInput.value === "string") {
        hiddenInput.value = nextCode;
        hiddenInput.dispatchEvent(new Event("input", { bubbles: true }));
        hiddenInput.dispatchEvent(new Event("change", { bubbles: true }));
      }
    },
    {
      nextCity: resolvedStation.stationName,
      nextCode: resolvedStation.stationCode,
      visibleSelector: inputSelector,
      hiddenSelector: hiddenCodeSelector
    }
  );

  return true;
}

async function fillStationField(
  page: Page,
  inputSelector: string,
  hiddenCodeSelector: string,
  city: string | null,
  preferHighSpeedStation = false
): Promise<boolean> {
  if (!city) {
    return false;
  }

  const stationTargets = buildStationSearchTargets(city, preferHighSpeedStation);
  const input = page.locator(inputSelector).first();
  await input.waitFor({
    state: "visible",
    timeout: 10000
  });

  await clearStationInput(page, input);
  await page.keyboard.type(city, {
    delay: CITY_INPUT_DELAY_MS
  });

  const selected = await clickCitySuggestion(page, stationTargets);
  await page.waitForTimeout(300);

  let visibleValue = normalizeText(await input.inputValue().catch(() => null));
  let hiddenCode = normalizeText(
    await page.locator(hiddenCodeSelector).first().inputValue().catch(() => null)
  );

  if ((!selected || !hiddenCode || !visibleValue) && city) {
    const fallbackApplied = await applyStationFallback(
      page,
      inputSelector,
      hiddenCodeSelector,
      stationTargets
    );

    if (fallbackApplied) {
      await page.waitForTimeout(200);
      visibleValue = normalizeText(await input.inputValue().catch(() => null));
      hiddenCode = normalizeText(
        await page.locator(hiddenCodeSelector).first().inputValue().catch(() => null)
      );
    }
  }

  return Boolean(hiddenCode) && Boolean(visibleValue);
}

async function fillTravelDate(page: Page, travelDate: string | null): Promise<boolean> {
  if (!travelDate) {
    return false;
  }

  const input = page.locator(TRAIN_DATE_SELECTOR).first();
  await input.waitFor({
    state: "attached",
    timeout: 10000
  });

  await input
    .evaluate((element, nextDate) => {
      const dateInput = element as {
        removeAttribute(name: string): void;
        value?: string;
        dispatchEvent(event: Event): boolean;
      };

      dateInput.removeAttribute("readonly");

      if (typeof dateInput.value === "string") {
        dateInput.value = nextDate;
      }

      dateInput.dispatchEvent(new Event("input", { bubbles: true }));
      dateInput.dispatchEvent(new Event("change", { bubbles: true }));
      dateInput.dispatchEvent(new Event("blur", { bubbles: true }));
    }, travelDate)
    .catch(() => undefined);

  await page.waitForTimeout(300);

  const currentValue = normalizeText(await input.inputValue().catch(() => null));
  return currentValue.startsWith(travelDate);
}

async function clickQueryButton(page: Page): Promise<void> {
  const button = page.locator(QUERY_BUTTON_SELECTOR).first();
  await button.waitFor({
    state: "visible",
    timeout: 10000
  });

  await button.click({
    timeout: 10000
  });
}

async function waitForVisibleResultRows(page: Page, timeoutMs = RESULT_WAIT_TIMEOUT_MS): Promise<void> {
  await page.locator(RESULT_TABLE_SELECTOR).first().waitFor({
    state: "visible",
    timeout: timeoutMs
  });

  const rows = page.locator(RESULT_ROW_SELECTOR);
  const startedAt = Date.now();

  while (Date.now() - startedAt < timeoutMs) {
    const rowCount = await rows.count();

    for (let index = 0; index < rowCount; index += 1) {
      const row = rows.nth(index);
      const isVisible = await row.isVisible().catch(() => false);

      if (isVisible) {
        return;
      }
    }

    await page.waitForTimeout(300);
  }

  throw new Error("12306 查询结果表未在预期时间内出现。");
}

async function getHeaderTexts(page: Page): Promise<string[]> {
  const headers = page.locator("th");
  const count = Math.min(await headers.count(), 40);
  const values: string[] = [];

  for (let index = 0; index < count; index += 1) {
    values.push(normalizeText(await headers.nth(index).textContent().catch(() => null)));
  }

  return values;
}

function resolveFirstClassCellIndex(headers: string[]): number | null {
  const exactMatchIndex = headers.findIndex((header) => header === "一等座");
  if (exactMatchIndex >= 4) {
    return exactMatchIndex - 3;
  }

  const looseMatchIndex = headers.findIndex(
    (header) => header.includes("一等座") && !header.includes("优选")
  );
  return looseMatchIndex >= 4 ? looseMatchIndex - 3 : null;
}

function resolveSecondClassCellIndex(headers: string[]): number | null {
  const headerIndex = headers.findIndex((header) => header.includes("二等座"));
  return headerIndex >= 4 ? headerIndex - 3 : null;
}

async function extractRowCellTexts(row: Locator): Promise<string[]> {
  const cells = row.locator("td");
  const count = await cells.count();
  const values: string[] = [];

  for (let index = 0; index < count; index += 1) {
    values.push(normalizeText(await cells.nth(index).textContent().catch(() => null)));
  }

  return values;
}

async function safeTextContent(locator: Locator, timeoutMs = 800): Promise<string | null> {
  const count = await locator.count().catch(() => 0);
  if (count === 0) {
    return null;
  }

  return locator
    .first()
    .textContent({
      timeout: timeoutMs
    })
    .catch(() => null);
}

async function safeAttribute(locator: Locator, name: string, timeoutMs = 800): Promise<string | null> {
  const count = await locator.count().catch(() => 0);
  if (count === 0) {
    return null;
  }

  return locator
    .first()
    .getAttribute(name, {
      timeout: timeoutMs
    })
    .catch(() => null);
}

function parseTimeToMinutes(value: string | null | undefined): number | null {
  const normalized = normalizeText(value);
  const match = normalized.match(/^(\d{1,2}):(\d{2})$/);
  if (!match) {
    return null;
  }

  const hours = Number(match[1]);
  const minutes = Number(match[2]);
  if (hours < 0 || hours > 23 || minutes < 0 || minutes > 59) {
    return null;
  }

  return hours * 60 + minutes;
}

function rankParsedRowsByIntent(parsedRows: ParsedTrainCandidate[], intent: BookingIntent): ParsedTrainCandidate[] {
  const lowerBound = parseTimeToMinutes(intent.departureTimeLowerBound);
  const upperBound = parseTimeToMinutes(intent.departureTimeUpperBound);

  const decorated = parsedRows.map((row, index) => {
    const departureMinutes = parseTimeToMinutes(row.departureTime);
    let timePenalty = 0;
    let priorityBucket = 3;

    if (departureMinutes !== null && lowerBound !== null && upperBound !== null) {
      if (departureMinutes >= lowerBound && departureMinutes <= upperBound) {
        priorityBucket = 0;
      } else {
        timePenalty = Math.min(Math.abs(departureMinutes - lowerBound), Math.abs(departureMinutes - upperBound));
      }
    } else if (departureMinutes !== null && lowerBound !== null) {
      if (departureMinutes >= lowerBound) {
        priorityBucket = 0;
        timePenalty = departureMinutes - lowerBound;
      } else {
        priorityBucket = 1;
        timePenalty = lowerBound - departureMinutes;
      }
    } else if (departureMinutes !== null && upperBound !== null) {
      if (departureMinutes <= upperBound) {
        priorityBucket = 0;
        timePenalty = upperBound - departureMinutes;
      } else {
        priorityBucket = 1;
        timePenalty = departureMinutes - upperBound;
      }
    }

    const trainTypePenalty =
      intent.trainTypes.length > 0 && !intent.trainTypes.includes(inferTrainType(row.trainNo)) ? 1 : 0;

    return {
      row,
      index,
      priorityBucket,
      timePenalty,
      trainTypePenalty
    };
  });

  decorated.sort((left, right) => {
    if (left.priorityBucket !== right.priorityBucket) {
      return left.priorityBucket - right.priorityBucket;
    }

    if (left.trainTypePenalty !== right.trainTypePenalty) {
      return left.trainTypePenalty - right.trainTypePenalty;
    }

    if (left.timePenalty !== right.timePenalty) {
      return left.timePenalty - right.timePenalty;
    }

    return left.index - right.index;
  });

  return decorated.slice(0, 3).map((item) => item.row);
}

async function parseVisibleTrainRows(page: Page, intent: BookingIntent): Promise<ParsedTrainCandidate[]> {
  const headers = await getHeaderTexts(page);
  const firstClassCellIndex = resolveFirstClassCellIndex(headers);
  const secondClassCellIndex = resolveSecondClassCellIndex(headers);
  const rows = page.locator(RESULT_ROW_SELECTOR);
  const rowCount = await rows.count();
  const parsed: ParsedTrainCandidate[] = [];

  for (let index = 0; index < rowCount; index += 1) {
    if (parsed.length >= 20) {
      break;
    }

    const row = rows.nth(index);
    const isVisible = await row.isVisible().catch(() => false);

    if (!isVisible) {
      continue;
    }

    const infoCell = row.locator("td").first();
    const trainNo = normalizeText(await safeTextContent(infoCell.locator("a.number")));
    const originStation =
      normalizeText(await safeAttribute(infoCell.locator(".start-s"), "title")) ||
      normalizeText(await safeTextContent(infoCell.locator(".start-s")));
    const destinationStation =
      normalizeText(await safeAttribute(infoCell.locator(".end-s"), "title")) ||
      normalizeText(await safeTextContent(infoCell.locator(".end-s")));
    const departureTime = normalizeText(await safeTextContent(infoCell.locator(".start-t")));
    const arrivalTime =
      normalizeText(await safeTextContent(infoCell.locator(".cds strong").nth(1))) ||
      normalizeText(await safeTextContent(infoCell.locator(".cds strong").last())) ||
      normalizeText(await safeTextContent(infoCell.locator(".color999")));
    const durationText =
      normalizeText(await safeTextContent(infoCell.locator(".ls strong"))) ||
      normalizeText(await safeTextContent(infoCell.locator(".ls")));

    const cellTexts = await extractRowCellTexts(row);
    const firstClassStatus =
      firstClassCellIndex !== null && firstClassCellIndex < cellTexts.length
        ? cellTexts[firstClassCellIndex]
        : "--";
    const secondClassStatus =
      secondClassCellIndex !== null && secondClassCellIndex < cellTexts.length
        ? cellTexts[secondClassCellIndex]
        : "--";

    if (!trainNo || !departureTime || !arrivalTime) {
      continue;
    }

    parsed.push({
      trainNo,
      originStation,
      destinationStation,
      departureTime,
      arrivalTime,
      durationText,
      firstClassStatus,
      secondClassStatus
    });
  }

  return rankParsedRowsByIntent(parsed, intent);
}

const PASSENGER_PAGE_URL_PATTERN = /confirmPassenger|confirmSingleForQueue/i;
const PAYMENT_PAGE_URL_PATTERN = /pay|cashier|payment|resultOrderForDcQueue|waitOrderComplete/i;
const PASSENGER_CONTAINER_SELECTORS = ["#normal_passenger_id", "ul#normal_passenger_id", "div#normal_passenger_id"];
const PASSENGER_LABEL_SELECTORS = [
  "#normal_passenger_id label",
  "#normal_passenger_id li label",
  "ul#normal_passenger_id label",
  "#normal_passenger_id li",
  "ul#normal_passenger_id li"
];
const PASSENGER_CHECKBOX_SELECTORS = [
  "#normal_passenger_id input[type='checkbox']",
  "ul#normal_passenger_id input[type='checkbox']",
  "input[type='checkbox'][id*='normalPassenger']",
  "input[type='checkbox'][name*='normalPassenger']"
];
const PASSENGER_READY_SELECTORS = [
  ...PASSENGER_LABEL_SELECTORS,
  ...PASSENGER_CHECKBOX_SELECTORS,
  ...PASSENGER_CONTAINER_SELECTORS,
  "#submitOrder_id",
  "select[id^='seatType_']",
  "#ticketInfo_id"
];
const SUBMIT_ORDER_TEXT = "\u63d0\u4ea4\u8ba2\u5355";
const SUBMIT_ORDER_SELECTORS = [
  "#submitOrder_id",
  `a:has-text("${SUBMIT_ORDER_TEXT}")`,
  `button:has-text("${SUBMIT_ORDER_TEXT}")`
];
const FINAL_CONFIRM_SELECTORS = [
  "#qr_submit_id",
  "a#qr_submit_id",
  "button#qr_submit_id",
  "a:has-text(\"\u786e\u8ba4\")",
  "button:has-text(\"\u786e\u8ba4\")",
  "a:has-text(\"\u786e\u5b9a\")",
  "button:has-text(\"\u786e\u5b9a\")"
];
const PAYMENT_READY_SELECTORS = [
  "#showCheckPay",
  "#payButton",
  "#continuePay",
  "#finishOrder_id",
  ".pay-tit",
  ".payment-hd"
];
const LOGIN_RESUME_TIMEOUT_MS = 120000;
const SUBMIT_CONFIRM_TIMEOUT_MS = 20000;
const PAYMENT_READY_TIMEOUT_MS = 45000;
const PASSENGER_TEXT_ALL = "\u5168\u90e8";

type ActiveTaskBrowserSession = BrowserPageSession & {
  taskId: ID;
  intent: BookingIntent;
  candidateTrainNo: string;
  seatType: string;
  passengerNames: string[];
  stage: "awaiting_login" | "order_page_ready" | "awaiting_payment";
  updatedAt: number;
};

type PassengerSelectionResult = {
  selectedPassengers: string[];
  fallbackUsed: boolean;
};

type SeatSelectionResult = {
  requestedSeat: string | null;
  matchedOption: string | null;
};

type PassengerOption = {
  checkbox: Locator;
  labelText: string;
};

async function findVisibleTrainRow(page: Page, candidateTrainNo: string): Promise<Locator | null> {
  const rows = page.locator(RESULT_ROW_SELECTOR);
  const rowCount = await rows.count();

  for (let index = 0; index < rowCount; index += 1) {
    const row = rows.nth(index);
    const isVisible = await row.isVisible().catch(() => false);

    if (!isVisible) {
      continue;
    }

    const trainNo = normalizeText(await row.locator("a.number").first().textContent().catch(() => null));
    if (trainNo === candidateTrainNo) {
      return row;
    }
  }

  return null;
}

async function clickPreorderButton(row: Locator): Promise<string> {
  const actionLinks = row.locator("a");
  const actionCount = await actionLinks.count();

  for (let index = 0; index < actionCount; index += 1) {
    const actionLink = actionLinks.nth(index);
    const isVisible = await actionLink.isVisible().catch(() => false);

    if (!isVisible) {
      continue;
    }

    const className = normalizeText(await actionLink.getAttribute("class").catch(() => null));
    const text = normalizeText(await actionLink.textContent().catch(() => null));

    if (className.includes("btn72") || text === PREORDER_TEXT) {
      await actionLink.click({
        timeout: 10000
      });
      return className.includes("btn72") ? ".btn72" : `text:${PREORDER_TEXT}`;
    }
  }

  throw new Error("12306 目标车次行中未找到可点击的预订按钮。");
}

async function waitForLoginInterception(
  page: Page,
  timeoutMs = 15000
): Promise<{ mode: string; currentUrl: string }> {
  const urlPromise = page
    .waitForURL((url) => /login|passport/i.test(url.toString()), {
      timeout: timeoutMs
    })
    .then(() => ({
      mode: "url_redirect",
      currentUrl: page.url()
    }))
    .catch(() => null);

  const selectorPromises = LOGIN_MODAL_SELECTORS.map((selector) =>
    page
      .locator(selector)
      .first()
      .waitFor({
        state: "visible",
        timeout: timeoutMs
      })
      .then(() => ({
        mode: `visible_selector:${selector}`,
        currentUrl: page.url()
      }))
      .catch(() => null)
  );

  const result = await Promise.race([
    urlPromise,
    ...selectorPromises,
    page.waitForTimeout(timeoutMs).then(() => null)
  ]);

  if (!result) {
    throw new Error("点击预订后未检测到 12306 登录拦截。");
  }

  return result;
}

async function waitForAnyVisibleSelector(
  page: Page,
  selectors: string[],
  timeoutMs: number
): Promise<string | null> {
  const selectorPromises = selectors.map((selector) =>
    page
      .locator(selector)
      .first()
      .waitFor({
        state: "visible",
        timeout: timeoutMs
      })
      .then(() => selector)
      .catch(() => null)
  );

  const result = await Promise.race([
    ...selectorPromises,
    page.waitForTimeout(timeoutMs).then(() => null)
  ]);

  return result ?? null;
}

async function getCurrentlyVisibleSelector(page: Page, selectors: string[]): Promise<string | null> {
  for (const selector of selectors) {
    const isVisible = await page.locator(selector).first().isVisible().catch(() => false);
    if (isVisible) {
      return selector;
    }
  }

  return null;
}

async function ensureTrainRowVisible(
  page: Page,
  intent: BookingIntent,
  candidateTrainNo: string,
  reporter?: BrowserProgressReporter
): Promise<Locator> {
  let targetRow = await findVisibleTrainRow(page, candidateTrainNo);
  if (targetRow) {
    return targetRow;
  }

  if (!/leftTicket/i.test(page.url())) {
    const navigationSucceeded = await gotoLeftTicketPage(page);
    if (!navigationSucceeded) {
      throw new Error("无法返回 12306 查票页以继续预订。");
    }

    await emitBrowserProgress(reporter, "URL", `[URL恢复] ${page.url()}`, {
      url: page.url(),
      candidateTrainNo
    });
  }

  await emitBrowserProgress(reporter, "TRACE", `[重试查票] 未找到 ${candidateTrainNo}，准备重新查询`, {
    candidateTrainNo
  });
  await runRealTrainSearch(page, intent, reporter);
  targetRow = await findVisibleTrainRow(page, candidateTrainNo);

  if (!targetRow) {
    throw new Error(`未在 12306 查询结果中找到车次 ${candidateTrainNo}。`);
  }

  return targetRow;
}

async function waitForPassengerPageReady(
  page: Page,
  timeoutMs: number
): Promise<{ mode: string; currentUrl: string } | null> {
  if (
    PASSENGER_PAGE_URL_PATTERN.test(page.url()) ||
    (await getCurrentlyVisibleSelector(page, PASSENGER_READY_SELECTORS))
  ) {
    return {
      mode: "already_ready",
      currentUrl: page.url()
    };
  }

  const navigationPromise = page
    .waitForNavigation({
      waitUntil: "domcontentloaded",
      timeout: timeoutMs
    })
    .then(() => {
      if (PASSENGER_PAGE_URL_PATTERN.test(page.url())) {
        return {
          mode: "navigation",
          currentUrl: page.url()
        };
      }

      return null;
    })
    .catch(() => null);

  const urlPromise = page
    .waitForURL((url) => PASSENGER_PAGE_URL_PATTERN.test(url.toString()), {
      timeout: timeoutMs
    })
    .then(() => ({
      mode: "url_confirm_passenger",
      currentUrl: page.url()
    }))
    .catch(() => null);

  const selectorPromise = waitForAnyVisibleSelector(page, PASSENGER_READY_SELECTORS, timeoutMs).then(
    (selector) =>
      selector
        ? {
            mode: `visible_selector:${selector}`,
            currentUrl: page.url()
          }
        : null
  );

  const result = await Promise.race([
    navigationPromise,
    urlPromise,
    selectorPromise,
    page.waitForTimeout(timeoutMs).then(() => null)
  ]);

  return result ?? null;
}

async function clickLocatorReliably(locator: Locator): Promise<void> {
  await locator.scrollIntoViewIfNeeded().catch(() => undefined);
  await locator.click({ timeout: 10000 }).catch(async () => {
    await locator.click({ force: true, timeout: 10000 }).catch(async () => {
      await locator.evaluate((node) => {
        const element = node as {
          click?: () => void;
        };
        element.click?.();
      });
    });
  });
}

async function clickFirstVisibleSelector(
  page: Page,
  selectors: string[],
  timeoutMs = 10000
): Promise<string | null> {
  const startedAt = Date.now();

  while (Date.now() - startedAt < timeoutMs) {
    for (const selector of selectors) {
      const locator = page.locator(selector).first();
      const isVisible = await locator.isVisible().catch(() => false);

      if (!isVisible) {
        continue;
      }

      await clickLocatorReliably(locator);
      return selector;
    }

    await page.waitForTimeout(300);
  }

  return null;
}

async function extractCheckboxText(checkbox: Locator): Promise<string> {
  return normalizeText(
    await checkbox
      .evaluate((node) => {
        const input = node as unknown as {
          closest?: (selector: string) => { textContent?: string | null } | null;
          parentElement?: { textContent?: string | null } | null;
        };

        return (
          input.closest?.("li,label,tr,div")?.textContent ??
          input.parentElement?.textContent ??
          ""
        );
      })
      .catch(() => null)
  );
}

async function ensureCheckboxSelected(checkbox: Locator): Promise<void> {
  const alreadyChecked = await checkbox.isChecked().catch(() => false);
  if (alreadyChecked) {
    return;
  }

  await checkbox.scrollIntoViewIfNeeded().catch(() => undefined);
  await checkbox.check({ force: true }).catch(async () => {
    await checkbox.click({ force: true, timeout: 10000 }).catch(async () => {
      const label = checkbox.locator("xpath=ancestor::label[1]").first();
      if (await label.isVisible().catch(() => false)) {
        await clickLocatorReliably(label);
      }
    });
  });

  const checkedAfterClick = await checkbox.isChecked().catch(() => false);
  if (checkedAfterClick) {
    return;
  }

  await checkbox.evaluate((node) => {
    const input = node as {
      checked?: boolean;
      dispatchEvent(event: Event): boolean;
      click?: () => void;
    };

    if (typeof input.checked === "boolean") {
      input.checked = true;
    }

    input.click?.();
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

async function waitForPassengerListReady(page: Page, timeoutMs = 15000): Promise<string | null> {
  await page.waitForLoadState("domcontentloaded", {
    timeout: Math.min(timeoutMs, 10000)
  }).catch(() => undefined);

  await page
    .locator("#normal_passenger_id label")
    .first()
    .waitFor({
      state: "visible",
      timeout: Math.min(timeoutMs, 15000)
    })
    .catch(() => undefined);

  const startedAt = Date.now();

  while (Date.now() - startedAt < timeoutMs) {
    const labelSelector = await getCurrentlyVisibleSelector(page, PASSENGER_LABEL_SELECTORS);
    if (labelSelector) {
      return labelSelector;
    }

    const checkboxSelector = await getCurrentlyVisibleSelector(page, PASSENGER_CHECKBOX_SELECTORS);
    if (checkboxSelector) {
      return checkboxSelector;
    }

    await page.waitForTimeout(400);
  }

  return null;
}

async function collectVisiblePassengerOptions(page: Page): Promise<PassengerOption[]> {
  const passengerOptions: PassengerOption[] = [];
  const preferredContainers = page.locator(
    "#normal_passenger_id li, #normal_passenger_id label, ul#normal_passenger_id li, ul#normal_passenger_id label"
  );
  const preferredCount = Math.min(await preferredContainers.count(), 60);

  for (let index = 0; index < preferredCount; index += 1) {
    const container = preferredContainers.nth(index);
    const isVisible = await container.isVisible().catch(() => false);

    if (!isVisible) {
      continue;
    }

    const checkbox = container.locator("input[type='checkbox']").first();
    if ((await checkbox.count()) === 0) {
      continue;
    }

    passengerOptions.push({
      checkbox,
      labelText: normalizeText(await container.textContent().catch(() => null)) || (await extractCheckboxText(checkbox))
    });
  }

  if (passengerOptions.length > 0) {
    return passengerOptions;
  }

  const fallbackCheckboxes = page.locator(
    [
      ...PASSENGER_CHECKBOX_SELECTORS,
      "#normal_passenger_id input[type='checkbox']",
      "input[type='checkbox']"
    ].join(", ")
  );
  const fallbackCount = Math.min(await fallbackCheckboxes.count(), 40);

  for (let index = 0; index < fallbackCount; index += 1) {
    const checkbox = fallbackCheckboxes.nth(index);
    const isVisible = await checkbox.isVisible().catch(() => false);

    if (!isVisible) {
      continue;
    }

    passengerOptions.push({
      checkbox,
      labelText: await extractCheckboxText(checkbox)
    });
  }

  return passengerOptions;
}

async function trySelectPassengerByName(page: Page, passengerName: string): Promise<string | null> {
  const normalizedTarget = normalizeCompactText(passengerName);
  const passengerOptions = await collectVisiblePassengerOptions(page);

  for (const passengerOption of passengerOptions) {
    const optionText = normalizeCompactText(passengerOption.labelText);
    if (!optionText || !optionText.includes(normalizedTarget)) {
      continue;
    }

    await ensureCheckboxSelected(passengerOption.checkbox);
    return normalizeText(passengerOption.labelText) || normalizeText(passengerName);
  }

  return null;
}

async function selectPassengers(
  page: Page,
  passengerNames: string[]
): Promise<PassengerSelectionResult> {
  const readySelector = await waitForPassengerListReady(page, 15000);
  if (!readySelector) {
    throw new Error("未找到 12306 乘客选择区域。");
  }

  const passengerOptions = await collectVisiblePassengerOptions(page);
  if (passengerOptions.length === 0) {
    throw new Error("未找到可选乘客，请先在 12306 中维护乘车人信息。");
  }

  const normalizedRequestedNames = passengerNames
    .map((name) => normalizeText(name))
    .filter((name) => Boolean(name) && name !== PASSENGER_TEXT_ALL);
  const selectedPassengers: string[] = [];

  for (const passengerName of normalizedRequestedNames) {
    const selectedName = await trySelectPassengerByName(page, passengerName);
    if (selectedName) {
      selectedPassengers.push(selectedName);
    }
  }

  if (selectedPassengers.length > 0) {
    return {
      selectedPassengers,
      fallbackUsed: false
    };
  }

  const firstVisiblePassenger = passengerOptions[0];
  await ensureCheckboxSelected(firstVisiblePassenger.checkbox);

  return {
    selectedPassengers: [normalizeText(firstVisiblePassenger.labelText) || "首位乘客"],
    fallbackUsed: true
  };
}

function resolveSeatAliases(seatPreferences: SeatType[]): string[] {
  const normalized = seatPreferences.map((seat) => normalizeText(seat)).filter(Boolean);
  const aliases = new Set<string>();

  for (const seat of normalized) {
    aliases.add(seat);

    if (seat.includes("二等")) {
      aliases.add("二等座");
      aliases.add("二等包座");
    }

    if (seat.includes("一等")) {
      aliases.add("一等座");
      aliases.add("优选一等座");
      aliases.add("一等卧");
    }

    if (seat.includes("商务")) {
      aliases.add("商务座");
    }

    if (seat.includes("特等")) {
      aliases.add("特等座");
    }

    if (seat.includes("无座")) {
      aliases.add("无座");
    }

    if (seat.includes("硬卧")) {
      aliases.add("硬卧");
      aliases.add("二等卧");
    }

    if (seat.includes("软卧")) {
      aliases.add("软卧");
      aliases.add("动卧");
    }
  }

  if (aliases.size === 0) {
    aliases.add("二等座");
  }

  return Array.from(aliases);
}

async function selectSeatPreference(
  page: Page,
  seatPreferences: SeatType[]
): Promise<SeatSelectionResult> {
  const requestedSeat = normalizeText(seatPreferences[0] ?? "") || null;
  const aliases = resolveSeatAliases(seatPreferences).map((seat) => normalizeCompactText(seat));
  const selectLocator = page
    .locator("select[id^='seatType_'], select[id*='seatType'], select[name*='seatType']")
    .first();

  await page.waitForTimeout(600);
  await selectLocator.waitFor({ state: "visible", timeout: 5000 }).catch(() => undefined);

  if (await selectLocator.isVisible().catch(() => false)) {
    const options = await selectLocator.locator("option").evaluateAll((nodes) =>
      nodes.map((node) => {
        const option = node as unknown as { value?: string; textContent?: string | null };
        return {
          value: option.value ?? "",
          text: (option.textContent ?? "").replace(/\s+/g, " ").trim()
        };
      })
    );

    const matchedOption = options.find((option) =>
      aliases.some((alias) => normalizeCompactText(option.text).includes(alias))
    );

    if (matchedOption?.value) {
      await selectLocator.selectOption(matchedOption.value);
      return {
        requestedSeat,
        matchedOption: matchedOption.text
      };
    }
  }

  return {
    requestedSeat,
    matchedOption: null
  };
}

async function waitForPaymentPageReady(
  page: Page,
  timeoutMs: number
): Promise<{ mode: string; currentUrl: string } | null> {
  if (
    PAYMENT_PAGE_URL_PATTERN.test(page.url()) ||
    (await getCurrentlyVisibleSelector(page, PAYMENT_READY_SELECTORS))
  ) {
    return {
      mode: "already_ready",
      currentUrl: page.url()
    };
  }

  const urlPromise = page
    .waitForURL((url) => PAYMENT_PAGE_URL_PATTERN.test(url.toString()), {
      timeout: timeoutMs
    })
    .then(() => ({
      mode: "url_payment",
      currentUrl: page.url()
    }))
    .catch(() => null);

  const selectorPromise = waitForAnyVisibleSelector(page, PAYMENT_READY_SELECTORS, timeoutMs).then(
    (selector) =>
      selector
        ? {
            mode: `visible_selector:${selector}`,
            currentUrl: page.url()
          }
        : null
  );

  const textPromise = page
    .waitForFunction(
      () => {
        const text = (globalThis as unknown as { document?: { body?: { innerText?: string } } }).document?.body?.innerText ?? "";
        return /\u652f\u4ed8|\u6536\u94f6\u53f0/.test(text);
      },
      {
        timeout: timeoutMs
      }
    )
    .then(() => ({
      mode: "body_text_payment",
      currentUrl: page.url()
    }))
    .catch(() => null);

  const result = await Promise.race([
    urlPromise,
    selectorPromise,
    textPromise,
    page.waitForTimeout(timeoutMs).then(() => null)
  ]);

  return result ?? null;
}

async function waitForSubmitOutcome(
  page: Page,
  timeoutMs: number
): Promise<
  | { kind: "confirm_dialog"; selector: string }
  | { kind: "payment_ready"; mode: string; currentUrl: string }
  | null
> {
  const confirmPromises = FINAL_CONFIRM_SELECTORS.map((selector) =>
    page
      .locator(selector)
      .first()
      .waitFor({
        state: "visible",
        timeout: timeoutMs
      })
      .then(() => ({
        kind: "confirm_dialog" as const,
        selector
      }))
      .catch(() => null)
  );

  const paymentPromise = waitForPaymentPageReady(page, timeoutMs).then((result) =>
    result
      ? {
          kind: "payment_ready" as const,
          mode: result.mode,
          currentUrl: result.currentUrl
        }
      : null
  );

  const result = await Promise.race([
    ...confirmPromises,
    paymentPromise,
    page.waitForTimeout(timeoutMs).then(() => null)
  ]);

  return result ?? null;
}

function mapToTrainCandidates(parsedRows: ParsedTrainCandidate[]): TrainCandidate[] {
  return parsedRows.map((candidate, index) => ({
    trainNo: candidate.trainNo,
    originStation: candidate.originStation,
    destinationStation: candidate.destinationStation,
    departureTime: candidate.departureTime as TrainCandidate["departureTime"],
    arrivalTime: candidate.arrivalTime as TrainCandidate["arrivalTime"],
    durationText: candidate.durationText,
    trainType: inferTrainType(candidate.trainNo),
    seatInventory: [
      buildSeatInventory("一等座", candidate.firstClassStatus),
      buildSeatInventory("二等座", candidate.secondClassStatus)
    ],
    score: calculateCandidateScore(candidate, index),
    isRecommended: index === 0
  }));
}

async function runRealTrainSearch(
  page: Page,
  intent: BookingIntent,
  reporter?: BrowserProgressReporter
): Promise<TrainCandidate[]> {
  const preferHighSpeedStation = intent.trainTypes.some((trainType) => ["G", "D", "C"].includes(trainType));
  await emitBrowserProgress(reporter, "DOM", `[命中节点] ${FROM_STATION_INPUT_SELECTOR}`, {
    selector: FROM_STATION_INPUT_SELECTOR,
    city: intent.origin ?? null
  });
  const originFilled = await fillStationField(
    page,
    FROM_STATION_INPUT_SELECTOR,
    FROM_STATION_CODE_SELECTOR,
    intent.origin,
    preferHighSpeedStation
  );

  if (!originFilled) {
    throw new Error("12306 出发地填写失败，未能从下拉列表中选中目标城市。");
  }

  await emitBrowserProgress(reporter, "OK", `[输入完成] 出发地 ${intent.origin ?? "未提供"}`, {
    selector: FROM_STATION_INPUT_SELECTOR,
    city: intent.origin ?? null
  });

  await emitBrowserProgress(reporter, "DOM", `[命中节点] ${TO_STATION_INPUT_SELECTOR}`, {
    selector: TO_STATION_INPUT_SELECTOR,
    city: intent.destination ?? null
  });
  const destinationFilled = await fillStationField(
    page,
    TO_STATION_INPUT_SELECTOR,
    TO_STATION_CODE_SELECTOR,
    intent.destination,
    preferHighSpeedStation
  );

  if (!destinationFilled) {
    throw new Error("12306 目的地填写失败，未能从下拉列表中选中目标城市。");
  }

  await emitBrowserProgress(reporter, "OK", `[输入完成] 目的地 ${intent.destination ?? "未提供"}`, {
    selector: TO_STATION_INPUT_SELECTOR,
    city: intent.destination ?? null
  });

  await emitBrowserProgress(reporter, "DOM", `[命中节点] ${TRAIN_DATE_SELECTOR}`, {
    selector: TRAIN_DATE_SELECTOR,
    travelDate: intent.travelDate ?? null
  });
  const dateFilled = await fillTravelDate(page, intent.travelDate);
  if (!dateFilled) {
    throw new Error("12306 出发日期填写失败。");
  }

  await emitBrowserProgress(reporter, "OK", `[输入完成] 出发日期 ${intent.travelDate ?? "未提供"}`, {
    selector: TRAIN_DATE_SELECTOR,
    travelDate: intent.travelDate ?? null
  });

  await emitBrowserProgress(reporter, "DOM", `[命中节点] ${QUERY_BUTTON_SELECTOR}`, {
    selector: QUERY_BUTTON_SELECTOR
  });
  await clickQueryButton(page);
  await waitForVisibleResultRows(page);
  await emitBrowserProgress(reporter, "DOM", `[命中节点] ${RESULT_TABLE_SELECTOR}`, {
    selector: RESULT_TABLE_SELECTOR
  });
  await page.waitForTimeout(SEARCH_VISUAL_WAIT_MS);

  const parsedRows = await parseVisibleTrainRows(page, intent);
  if (parsedRows.length === 0) {
    throw new Error("12306 页面已返回结果表，但未解析到有效车次数据。");
  }

  await emitBrowserProgress(
    reporter,
    "DATA",
    `[读取结果] 真实车次 ${parsedRows.length} 条：${parsedRows.map((row) => row.trainNo).join(", ")}`,
    {
      count: parsedRows.length,
      trainNos: parsedRows.map((row) => row.trainNo)
    }
  );

  return mapToTrainCandidates(parsedRows);
}

async function openLeftTicketPage(waitMs: number): Promise<{ navigationSucceeded: boolean }> {
  const session = await createBrowserPage();
  let navigationSucceeded = false;

  try {
    navigationSucceeded = await gotoLeftTicketPage(session.page);
    await session.page.waitForTimeout(waitMs);
  } finally {
    await closeBrowserPage(session);
  }

  return {
    navigationSucceeded
  };
}

export interface BrowserService {
  execute(request: BrowserActionRequest): Promise<BrowserActionResult>;
}

export class PlaywrightBrowserService implements BrowserService {
  private readonly humanVerifiedTasks = new Set<ID>();
  private readonly cachedSearchIntents = new Map<ID, BookingIntent>();
  private readonly activeTaskSessions = new Map<ID, ActiveTaskBrowserSession>();

  hasPendingLoginSession(taskId: ID): boolean {
    return this.activeTaskSessions.get(taskId)?.stage === "awaiting_login";
  }

  markHumanVerified(taskId: ID): void {
    this.humanVerifiedTasks.add(taskId);
  }

  private async replaceTaskSession(taskSession: ActiveTaskBrowserSession): Promise<void> {
    await this.closeTaskSession(taskSession.taskId);
    this.activeTaskSessions.set(taskSession.taskId, taskSession);
  }

  private async closeTaskSession(taskId: ID): Promise<void> {
    const existingSession = this.activeTaskSessions.get(taskId);
    if (!existingSession) {
      return;
    }

    this.activeTaskSessions.delete(taskId);
    await closeBrowserPage(existingSession).catch(() => undefined);
  }

  private getTaskSession(taskId: ID): ActiveTaskBrowserSession | null {
    return this.activeTaskSessions.get(taskId) ?? null;
  }

  async execute(request: BrowserActionRequest): Promise<BrowserActionResult> {
    switch (request.action) {
      case BrowserActionType.SEARCH_TRAINS:
        this.cachedSearchIntents.set(request.taskId, request.payload.intent);
        return this.searchTrains(request.payload.intent, request.onProgress);
      case BrowserActionType.WAIT_LOGIN:
        return this.waitLogin(request.taskId, request.payload.timeoutMs, request.onProgress);
      case BrowserActionType.SELECT_TRAIN:
        return this.selectTrain(
          request.taskId,
          request.payload.intent,
          request.payload.candidateTrainNo,
          request.payload.seatType,
          request.payload.passengerNames,
          request.onProgress
        );
      case BrowserActionType.SUBMIT_ORDER:
        return this.submitOrder(request.taskId, request.onProgress);
      default:
        return {
          ok: false,
          state: "failed",
          errorCode: ErrorCode.INTERNAL_ERROR,
          errorMessage: `Unsupported browser action: ${request.action}`
        };
    }
  }

  private async searchTrains(
    intent: BookingIntent,
    onProgress?: BrowserProgressReporter
  ): Promise<BrowserActionResult> {
    const session = await createBrowserPage();
    let navigationSucceeded = false;

    try {
      await emitBrowserProgress(onProgress, "TRACE", "[浏览器启动] 已创建新的 Chromium 会话", {
        stage: "search_trains"
      });
      navigationSucceeded = await gotoLeftTicketPage(session.page);

      if (!navigationSucceeded) {
        throw new Error("12306 查票页打开失败。");
      }

      await emitBrowserProgress(onProgress, "URL", `[URL跳转] ${session.page.url()}`, {
        url: session.page.url()
      });

      const candidates = await runRealTrainSearch(session.page, intent, onProgress);
      const data: BrowserSearchTrainsResultData = {
        candidates,
        openedUrl: session.page.url(),
        navigationSucceeded
      };

      await emitBrowserProgress(onProgress, "OK", `[查票完成] 已返回 ${candidates.length} 条候选车次`, {
        count: candidates.length,
        trainNos: candidates.map((candidate) => candidate.trainNo)
      });

      return {
        ok: true,
        state: "search_results_ready",
        data
      };
    } catch (error) {
      return {
        ok: false,
        state: "failed",
        errorCode: ErrorCode.INTERNAL_ERROR,
        errorMessage: getErrorMessage(error)
      };
    } finally {
      await closeBrowserPage(session);
    }
  }

  private async waitLogin(
    taskId: ID,
    timeoutMs: number,
    onProgress?: BrowserProgressReporter
  ): Promise<BrowserActionResult> {
    const taskSession = this.getTaskSession(taskId);

    if (!taskSession) {
      return {
        ok: false,
        state: "failed",
        errorCode: ErrorCode.INTERNAL_ERROR,
        errorMessage: "No active login session found for this task."
      };
    }

    if (!this.humanVerifiedTasks.has(taskId)) {
      await emitBrowserProgress(onProgress, "TRACE", "[人工接管] 等待用户在前端点击“已扫码登录”", {
        url: taskSession.page.url(),
        waitStage: "awaiting_human_signal"
      });
      return {
        ok: true,
        state: "waiting_human",
        data: {
          openedUrl: taskSession.page.url(),
          waitStage: "awaiting_human_signal"
        }
      };
    }

    try {
      await emitBrowserProgress(onProgress, "TRACE", "[恢复执行] 收到人工验证完成信号，开始检测登录结果", {
        timeoutMs,
        url: taskSession.page.url()
      });
      let passengerReady = await waitForPassengerPageReady(taskSession.page, timeoutMs);

      if (!passengerReady) {
        const stillVisibleLoginSelector = await getCurrentlyVisibleSelector(
          taskSession.page,
          LOGIN_MODAL_SELECTORS
        );

        if (stillVisibleLoginSelector) {
          this.humanVerifiedTasks.delete(taskId);
          await emitBrowserProgress(
            onProgress,
            "WARN",
            `[登录拦截] 登录界面仍然可见：${stillVisibleLoginSelector}`,
            {
              visibleSelector: stillVisibleLoginSelector,
              url: taskSession.page.url()
            }
          );
          return {
            ok: true,
            state: "waiting_human",
            data: {
              openedUrl: taskSession.page.url(),
              waitStage: "login_modal_still_visible",
              visibleSelector: stillVisibleLoginSelector
            }
          };
        }

        const targetRow = await ensureTrainRowVisible(
          taskSession.page,
          taskSession.intent,
          taskSession.candidateTrainNo,
          onProgress
        );
        await emitBrowserProgress(onProgress, "HIT", `[命中车次] ${taskSession.candidateTrainNo}`, {
          candidateTrainNo: taskSession.candidateTrainNo
        });
        const preorderSelector = await clickPreorderButton(targetRow);
        await emitBrowserProgress(onProgress, "DOM", `[命中节点] ${preorderSelector}`, {
          selector: preorderSelector,
          candidateTrainNo: taskSession.candidateTrainNo
        });
        passengerReady = await waitForPassengerPageReady(taskSession.page, Math.max(30000, timeoutMs / 2));
      }

      if (!passengerReady) {
        this.humanVerifiedTasks.delete(taskId);
        await emitBrowserProgress(onProgress, "WARN", "[等待乘客页] 未进入乘客确认页，仍需人工继续处理", {
          url: taskSession.page.url(),
          waitStage: "confirm_passenger_not_ready"
        });
        return {
          ok: true,
          state: "waiting_human",
          data: {
            openedUrl: taskSession.page.url(),
            waitStage: "confirm_passenger_not_ready"
          }
        };
      }

      await emitBrowserProgress(onProgress, "URL", `[URL恢复] ${passengerReady.currentUrl}`, {
        mode: passengerReady.mode,
        url: passengerReady.currentUrl
      });

      const passengerSelection = await selectPassengers(taskSession.page, taskSession.passengerNames);
      await emitBrowserProgress(
        onProgress,
        "HIT",
        `[乘客匹配] ${passengerSelection.selectedPassengers.join(", ") || "默认首位乘客"}`,
        {
          selectedPassengers: passengerSelection.selectedPassengers,
          fallbackPassengerUsed: passengerSelection.fallbackUsed
        }
      );
      const seatSelection = await selectSeatPreference(taskSession.page, taskSession.intent.seatPreferences);
      await emitBrowserProgress(
        onProgress,
        "DATA",
        `[席别匹配] ${seatSelection.requestedSeat ?? "默认席别"} -> ${seatSelection.matchedOption ?? "保持页面默认"}`,
        {
          requestedSeat: seatSelection.requestedSeat,
          matchedOption: seatSelection.matchedOption
        }
      );

      taskSession.stage = "order_page_ready";
      taskSession.updatedAt = Date.now();
      this.humanVerifiedTasks.delete(taskId);

      await emitBrowserProgress(onProgress, "OK", "[乘客页就绪] 已完成乘客与席别处理，等待最终提交", {
        url: taskSession.page.url()
      });

      return {
        ok: true,
        state: "order_page_ready",
        data: {
          openedUrl: taskSession.page.url(),
          resumeMode: passengerReady.mode,
          selectedPassengers: passengerSelection.selectedPassengers,
          fallbackPassengerUsed: passengerSelection.fallbackUsed,
          seatSelection
        }
      };
    } catch (error) {
      this.humanVerifiedTasks.delete(taskId);
      return {
        ok: false,
        state: "failed",
        errorCode: ErrorCode.INTERNAL_ERROR,
        errorMessage: getErrorMessage(error)
      };
    }
  }

  private async selectTrain(
    taskId: ID,
    intent: BookingIntent,
    candidateTrainNo: string,
    seatType: string,
    passengerNames: string[],
    onProgress?: BrowserProgressReporter
  ): Promise<BrowserActionResult> {
    this.cachedSearchIntents.set(taskId, intent);

    if (this.humanVerifiedTasks.has(taskId) || this.hasPendingLoginSession(taskId)) {
      return this.waitLogin(taskId, LOGIN_RESUME_TIMEOUT_MS, onProgress);
    }

    const session = await createBrowserPage();
    let navigationSucceeded = false;
    let keepSessionOpen = false;

    try {
      await emitBrowserProgress(onProgress, "TRACE", "[浏览器启动] 已创建新的 Chromium 会话", {
        stage: "select_train",
        candidateTrainNo
      });
      navigationSucceeded = await gotoLeftTicketPage(session.page);

      if (!navigationSucceeded) {
        throw new Error("12306 查票页打开失败，无法执行预订点击。");
      }

      await emitBrowserProgress(onProgress, "URL", `[URL跳转] ${session.page.url()}`, {
        url: session.page.url()
      });

      await runRealTrainSearch(session.page, intent, onProgress);

      const targetRow = await findVisibleTrainRow(session.page, candidateTrainNo);
      if (!targetRow) {
        throw new Error(`未在 12306 查询结果中找到车次 ${candidateTrainNo}。`);
      }

      await emitBrowserProgress(onProgress, "HIT", `[命中车次] ${candidateTrainNo}`, {
        candidateTrainNo,
        seatType
      });
      const preorderSelector = await clickPreorderButton(targetRow);
      await emitBrowserProgress(onProgress, "DOM", `[命中节点] ${preorderSelector}`, {
        selector: preorderSelector,
        candidateTrainNo
      });
      const interception = await waitForLoginInterception(session.page);
      await emitBrowserProgress(onProgress, "WARN", `[登录拦截] ${interception.mode}`, {
        interceptionMode: interception.mode,
        url: interception.currentUrl,
        candidateTrainNo
      });
      await session.page.waitForTimeout(DEFAULT_VISUAL_WAIT_MS);

      await this.replaceTaskSession({
        ...session,
        taskId,
        intent,
        candidateTrainNo,
        seatType,
        passengerNames,
        stage: "awaiting_login",
        updatedAt: Date.now()
      });
      keepSessionOpen = true;

      await emitBrowserProgress(onProgress, "OK", "[会话保留] 已保持浏览器现场，等待人工扫码登录", {
        url: interception.currentUrl,
        candidateTrainNo
      });

      return {
        ok: true,
        state: "login_required",
        data: {
          openedUrl: interception.currentUrl,
          candidateTrainNo,
          seatType,
          navigationSucceeded,
          interceptionMode: interception.mode
        }
      };
    } catch (error) {
      return {
        ok: false,
        state: "failed",
        errorCode: ErrorCode.INTERNAL_ERROR,
        errorMessage: getErrorMessage(error)
      };
    } finally {
      if (!keepSessionOpen) {
        await closeBrowserPage(session).catch(() => undefined);
      }
    }
  }

  private async submitOrder(taskId: ID, onProgress?: BrowserProgressReporter): Promise<BrowserActionResult> {
    const taskSession = this.getTaskSession(taskId);

    if (!taskSession) {
      return {
        ok: false,
        state: "failed",
        errorCode: ErrorCode.INTERNAL_ERROR,
        errorMessage: "No active order session found for this task."
      };
    }

    try {
      const passengerReady = await waitForPassengerPageReady(taskSession.page, 15000);
      if (!passengerReady) {
        throw new Error("未处于可提交订单的乘客确认页。");
      }

      await emitBrowserProgress(onProgress, "URL", `[URL检测] ${passengerReady.currentUrl}`, {
        mode: passengerReady.mode,
        url: passengerReady.currentUrl
      });

      const submitSelector = await clickFirstVisibleSelector(taskSession.page, SUBMIT_ORDER_SELECTORS);
      if (!submitSelector) {
        throw new Error("未找到 12306 提交订单按钮。");
      }

      await emitBrowserProgress(onProgress, "DOM", `[命中节点] ${submitSelector}`, {
        selector: submitSelector
      });

      let submitOutcome = await waitForSubmitOutcome(taskSession.page, SUBMIT_CONFIRM_TIMEOUT_MS);
      if (!submitOutcome) {
        throw new Error("点击提交订单后未等到确认弹窗或支付页信号。");
      }

      if (submitOutcome.kind === "confirm_dialog") {
        await emitBrowserProgress(onProgress, "DOM", `[命中节点] ${submitOutcome.selector}`, {
          selector: submitOutcome.selector
        });
        const confirmClickedSelector = await clickFirstVisibleSelector(taskSession.page, [
          submitOutcome.selector,
          ...FINAL_CONFIRM_SELECTORS
        ]);

        if (!confirmClickedSelector) {
          throw new Error("未找到核对信息确认按钮。");
        }

        await emitBrowserProgress(onProgress, "DOM", `[命中节点] ${confirmClickedSelector}`, {
          selector: confirmClickedSelector
        });

        const paymentReady = await waitForPaymentPageReady(taskSession.page, PAYMENT_READY_TIMEOUT_MS);
        if (!paymentReady) {
          throw new Error("提交订单后未跳转到支付页。");
        }

        await emitBrowserProgress(onProgress, "URL", `[支付页检测] ${paymentReady.currentUrl}`, {
          mode: paymentReady.mode,
          url: paymentReady.currentUrl
        });

        taskSession.stage = "awaiting_payment";
        taskSession.updatedAt = Date.now();

        await emitBrowserProgress(onProgress, "OK", "[提交流程完成] 已进入待支付状态", {
          url: paymentReady.currentUrl
        });

        return {
          ok: true,
          state: "order_submitted",
          data: {
            openedUrl: paymentReady.currentUrl,
            submitSelector,
            confirmSelector: confirmClickedSelector,
            paymentMode: paymentReady.mode
          }
        };
      }

      taskSession.stage = "awaiting_payment";
      taskSession.updatedAt = Date.now();

      await emitBrowserProgress(onProgress, "URL", `[支付页检测] ${submitOutcome.currentUrl}`, {
        mode: submitOutcome.mode,
        url: submitOutcome.currentUrl
      });
      await emitBrowserProgress(onProgress, "OK", "[提交流程完成] 已直接进入待支付状态", {
        url: submitOutcome.currentUrl
      });

      return {
        ok: true,
        state: "order_submitted",
        data: {
          openedUrl: submitOutcome.currentUrl,
          submitSelector,
          paymentMode: submitOutcome.mode
        }
      };
    } catch (error) {
      return {
        ok: false,
        state: "failed",
        errorCode: ErrorCode.INTERNAL_ERROR,
        errorMessage: getErrorMessage(error)
      };
    }
  }
}

export const browserService = new PlaywrightBrowserService();


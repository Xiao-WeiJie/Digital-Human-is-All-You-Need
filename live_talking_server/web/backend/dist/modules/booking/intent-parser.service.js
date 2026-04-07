"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.mockParseIntent = mockParseIntent;
const DEFAULT_SEAT_TYPE = "二等座";
const SEAT_TYPE_RULES = [
    { seatType: "商务座", keywords: ["商务座", "商务"] },
    { seatType: "特等座", keywords: ["特等座", "特等"] },
    { seatType: "一等座", keywords: ["一等座", "一等"] },
    { seatType: "二等座", keywords: ["二等座", "二等"] },
    { seatType: "高级软卧", keywords: ["高级软卧"] },
    { seatType: "软卧", keywords: ["软卧", "动卧"] },
    { seatType: "硬卧", keywords: ["硬卧"] },
    { seatType: "软座", keywords: ["软座"] },
    { seatType: "硬座", keywords: ["硬座"] },
    { seatType: "无座", keywords: ["无座"] }
];
const RELATIVE_DAY_OFFSETS = [
    { keyword: "大后天", days: 3 },
    { keyword: "后天", days: 2 },
    { keyword: "明天", days: 1 },
    { keyword: "今天", days: 0 },
    { keyword: "今日", days: 0 }
];
const WEEKDAY_INDEX = {
    一: 1,
    二: 2,
    三: 3,
    四: 4,
    五: 5,
    六: 6,
    日: 7,
    天: 7
};
function normalizeDate(year, month, day) {
    return `${String(year).padStart(4, "0")}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}
function padTime(value) {
    return String(value).padStart(2, "0");
}
function normalizeTime(hours, minutes) {
    if (!Number.isInteger(hours) ||
        !Number.isInteger(minutes) ||
        hours < 0 ||
        hours > 23 ||
        minutes < 0 ||
        minutes > 59) {
        return null;
    }
    return `${padTime(hours)}:${padTime(minutes)}`;
}
function buildValidDate(year, month, day) {
    const date = new Date(year, month - 1, day);
    if (date.getFullYear() !== year ||
        date.getMonth() !== month - 1 ||
        date.getDate() !== day) {
        return null;
    }
    date.setHours(0, 0, 0, 0);
    return date;
}
function startOfDay(date) {
    const next = new Date(date);
    next.setHours(0, 0, 0, 0);
    return next;
}
function addDays(date, days) {
    const next = new Date(date);
    next.setDate(next.getDate() + days);
    return next;
}
function toIsoDate(date) {
    return normalizeDate(date.getFullYear(), date.getMonth() + 1, date.getDate());
}
function resolveMonthDayDate(month, day, now) {
    const today = startOfDay(now);
    const currentYearCandidate = buildValidDate(today.getFullYear(), month, day);
    if (currentYearCandidate && currentYearCandidate >= today) {
        return toIsoDate(currentYearCandidate);
    }
    const nextYearCandidate = buildValidDate(today.getFullYear() + 1, month, day);
    return nextYearCandidate ? toIsoDate(nextYearCandidate) : null;
}
function getMonday(date) {
    const monday = startOfDay(date);
    const weekDay = monday.getDay();
    const delta = weekDay === 0 ? -6 : 1 - weekDay;
    monday.setDate(monday.getDate() + delta);
    return monday;
}
function resolveWeekdayDate(prefix, weekday, now) {
    const targetWeekday = WEEKDAY_INDEX[weekday];
    if (!targetWeekday) {
        return null;
    }
    const monday = getMonday(now);
    const weekOffset = prefix === "下下周" ? 2 : prefix === "下周" ? 1 : 0;
    monday.setDate(monday.getDate() + weekOffset * 7);
    const candidate = addDays(monday, targetWeekday - 1);
    if (weekOffset === 0 && candidate < startOfDay(now)) {
        candidate.setDate(candidate.getDate() + 7);
    }
    return toIsoDate(candidate);
}
function extractTravelDate(rawText, now = new Date()) {
    const normalizedText = rawText.replace(/\s+/g, "");
    const fullDateMatch = normalizedText.match(/(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})日?/) ??
        normalizedText.match(/(\d{4})(\d{2})(\d{2})/);
    if (fullDateMatch) {
        const year = Number(fullDateMatch[1]);
        const month = Number(fullDateMatch[2]);
        const day = Number(fullDateMatch[3]);
        const explicitDate = buildValidDate(year, month, day);
        return explicitDate ? toIsoDate(explicitDate) : null;
    }
    for (const { keyword, days } of RELATIVE_DAY_OFFSETS) {
        if (normalizedText.includes(keyword)) {
            return toIsoDate(addDays(startOfDay(now), days));
        }
    }
    const monthDayMatch = normalizedText.match(/(\d{1,2})月(\d{1,2})[日号]?/) ??
        normalizedText.match(/(\d{1,2})\/(\d{1,2})(?!\d)/) ??
        normalizedText.match(/(\d{1,2})\.(\d{1,2})(?!\d)/);
    if (monthDayMatch) {
        return resolveMonthDayDate(Number(monthDayMatch[1]), Number(monthDayMatch[2]), now);
    }
    const weekdayMatch = normalizedText.match(/(下下周|下周|本周|这周|周|星期)([一二三四五六日天])/);
    if (weekdayMatch) {
        return resolveWeekdayDate(weekdayMatch[1], weekdayMatch[2], now);
    }
    return null;
}
function normalizeLocation(value) {
    if (!value) {
        return null;
    }
    return value
        .replace(/(?:的)?(?:商务座|特等座|一等座|二等座|高级软卧|软卧|硬卧|软座|硬座|无座|高铁|动车|火车|列车|车票|票|班次|出发|发车).*/u, "")
        .replace(/[，。,.；;、\s].*$/, "")
        .trim() || null;
}
function extractRoute(rawText) {
    const routePatterns = [
        /从\s*(?<origin>[\u4E00-\u9FFF·A-Za-z]{1,20}?)\s*(?:站)?\s*(?:到|去)\s*(?<destination>[\u4E00-\u9FFF·A-Za-z]{1,20}?)(?:站)?(?=(?:的?(?:高铁|动车|火车|列车|车票|票|班次|出发|发车))|[，。,\.；;\s]|$)/u,
        /(?<origin>[\u4E00-\u9FFF·A-Za-z]{2,20}?)\s*(?:站)?\s*(?:到|去)\s*(?<destination>[\u4E00-\u9FFF·A-Za-z]{2,20}?)(?:站)?(?=(?:的?(?:高铁|动车|火车|列车|车票|票|班次|出发|发车))|[，。,\.；;\s]|$)/u
    ];
    for (const pattern of routePatterns) {
        const match = rawText.match(pattern);
        if (!match?.groups) {
            continue;
        }
        return {
            origin: normalizeLocation(match.groups.origin),
            destination: normalizeLocation(match.groups.destination)
        };
    }
    return {
        origin: null,
        destination: null
    };
}
function extractSeatPreferences(rawText) {
    const matchedSeatTypes = [];
    for (const rule of SEAT_TYPE_RULES) {
        if (rule.keywords.some((keyword) => rawText.includes(keyword))) {
            matchedSeatTypes.push(rule.seatType);
        }
    }
    return matchedSeatTypes.length > 0 ? matchedSeatTypes : [DEFAULT_SEAT_TYPE];
}
function extractTrainTypes(rawText) {
    const trainTypes = new Set();
    if (/高铁|G\d+/iu.test(rawText)) {
        trainTypes.add("G");
    }
    if (/动车|D\d+/iu.test(rawText)) {
        trainTypes.add("D");
    }
    if (/城际|C\d+/iu.test(rawText)) {
        trainTypes.add("C");
    }
    if (/直达|Z\d+/iu.test(rawText)) {
        trainTypes.add("Z");
    }
    if (/特快|T\d+/iu.test(rawText)) {
        trainTypes.add("T");
    }
    if (/快速|K\d+/iu.test(rawText)) {
        trainTypes.add("K");
    }
    return Array.from(trainTypes);
}
function applyDayPeriod(hours, period) {
    if (!period) {
        return hours;
    }
    switch (period) {
        case "凌晨":
            return hours === 12 ? 0 : hours;
        case "早上":
        case "上午":
            return hours === 12 ? 0 : hours;
        case "中午":
            return hours >= 11 ? hours : hours + 12;
        case "下午":
        case "傍晚":
        case "晚上":
        case "今晚":
        case "夜里":
        case "夜间":
            return hours >= 12 ? hours : hours + 12;
        default:
            return hours;
    }
}
function extractExactDepartureTime(rawText) {
    const exactClockMatch = rawText.match(/(?:凌晨|早上|上午|中午|下午|傍晚|晚上|今晚|夜里|夜间)?\s*(\d{1,2})[:：](\d{1,2})/);
    if (exactClockMatch) {
        const prefixMatch = rawText.match(/(凌晨|早上|上午|中午|下午|傍晚|晚上|今晚|夜里|夜间)\s*\d{1,2}[:：]\d{1,2}/);
        const period = prefixMatch?.[1] ?? null;
        const hours = applyDayPeriod(Number(exactClockMatch[1]), period);
        return normalizeTime(hours, Number(exactClockMatch[2]));
    }
    const chineseClockMatch = rawText.match(/(凌晨|早上|上午|中午|下午|傍晚|晚上|今晚|夜里|夜间)?\s*(\d{1,2})\s*(?:点|时)(半|一刻|三刻|(\d{1,2})分?)?/);
    if (!chineseClockMatch) {
        return null;
    }
    const period = chineseClockMatch[1] ?? null;
    const rawHours = Number(chineseClockMatch[2]);
    const minuteToken = chineseClockMatch[3] ?? null;
    const rawMinutes = chineseClockMatch[4] ?? null;
    let minutes = 0;
    if (minuteToken === "半") {
        minutes = 30;
    }
    else if (minuteToken === "一刻") {
        minutes = 15;
    }
    else if (minuteToken === "三刻") {
        minutes = 45;
    }
    else if (rawMinutes) {
        minutes = Number(rawMinutes);
    }
    const hours = applyDayPeriod(rawHours, period);
    return normalizeTime(hours, minutes);
}
function extractDepartureWindow(rawText) {
    const exactTime = extractExactDepartureTime(rawText);
    if (exactTime) {
        return {
            departureTimeLowerBound: exactTime,
            departureTimeUpperBound: null
        };
    }
    if (/凌晨/.test(rawText)) {
        return {
            departureTimeLowerBound: "00:00",
            departureTimeUpperBound: "05:59"
        };
    }
    if (/早上|上午/.test(rawText)) {
        return {
            departureTimeLowerBound: "06:00",
            departureTimeUpperBound: "11:59"
        };
    }
    if (/中午/.test(rawText)) {
        return {
            departureTimeLowerBound: "11:00",
            departureTimeUpperBound: "13:59"
        };
    }
    if (/下午/.test(rawText)) {
        return {
            departureTimeLowerBound: "13:00",
            departureTimeUpperBound: "17:59"
        };
    }
    if (/傍晚|晚上|今晚|夜里|夜间/.test(rawText)) {
        return {
            departureTimeLowerBound: "18:00",
            departureTimeUpperBound: "23:59"
        };
    }
    return {
        departureTimeLowerBound: null,
        departureTimeUpperBound: null
    };
}
function collectMissingSlots(intent) {
    const missingSlots = [];
    if (!intent.travelDate) {
        missingSlots.push("travelDate");
    }
    if (!intent.origin) {
        missingSlots.push("origin");
    }
    if (!intent.destination) {
        missingSlots.push("destination");
    }
    return missingSlots;
}
function mockParseIntent(rawText, taskId) {
    const { origin, destination } = extractRoute(rawText);
    const travelDate = extractTravelDate(rawText);
    const departureWindow = extractDepartureWindow(rawText);
    const trainTypes = extractTrainTypes(rawText);
    const intent = {
        rawText,
        travelDate,
        origin,
        destination,
        departureTimeLowerBound: departureWindow.departureTimeLowerBound,
        departureTimeUpperBound: departureWindow.departureTimeUpperBound,
        trainTypes,
        seatPreferences: extractSeatPreferences(rawText),
        passengerNames: [],
        allowWaitlist: /候补|无票也行|抢票/u.test(rawText),
        allowFallbackTime: /时间灵活|前后都可以|临近班次也可以/u.test(rawText),
        allowFallbackSeat: /席别灵活|无座也行|座位不限/u.test(rawText),
        queryOnly: /只查|查询|查一下|查下|帮我查|看一下|看看/u.test(rawText)
    };
    const missingSlots = collectMissingSlots(intent);
    const context = {
        taskId,
        turns: 1,
        missingSlots,
        ambiguousSlots: [],
        slotConfidence: [],
        lastClarificationQuestion: null
    };
    return {
        intent,
        context,
        needsClarification: missingSlots.length > 0
    };
}

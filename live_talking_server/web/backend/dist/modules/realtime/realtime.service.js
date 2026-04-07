"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.realtimeService = exports.InMemoryRealtimeService = void 0;
const task_repository_1 = require("../tasks/task.repository");
function formatSseEvent(event) {
    const payload = JSON.stringify({
        taskId: event.taskId,
        sequenceId: event.sequenceId,
        timestamp: event.createdAt,
        type: event.type,
        status: event.status,
        checkpointId: event.checkpointId,
        message: event.message,
        requiresAck: event.requiresAck,
        payload: event.payload
    });
    return `id: ${event.sequenceId}\nevent: task_event\ndata: ${payload}\n\n`;
}
class InMemoryRealtimeService {
    repository;
    subscribers = new Map();
    constructor(repository = task_repository_1.inMemoryTaskRepository) {
        this.repository = repository;
    }
    async publishTaskEvent(taskId, event) {
        const clients = this.subscribers.get(taskId);
        if (!clients || clients.size === 0) {
            return;
        }
        const frame = formatSseEvent(event);
        const brokenClients = [];
        for (const client of clients) {
            try {
                if ("destroyed" in client && client.destroyed) {
                    brokenClients.push(client);
                    continue;
                }
                client.write(frame);
            }
            catch {
                brokenClients.push(client);
            }
        }
        for (const brokenClient of brokenClients) {
            this.unsubscribe(taskId, brokenClient);
        }
    }
    async getEventsAfter(taskId, afterSequence) {
        const events = await this.repository.listEvents(taskId);
        return events.filter((event) => event.sequenceId > afterSequence);
    }
    async subscribeSSE(taskId, response) {
        const clients = this.subscribers.get(taskId) ?? new Set();
        clients.add(response);
        this.subscribers.set(taskId, clients);
    }
    unsubscribe(taskId, response) {
        const clients = this.subscribers.get(taskId);
        if (!clients) {
            return;
        }
        clients.delete(response);
        if (clients.size === 0) {
            this.subscribers.delete(taskId);
        }
    }
}
exports.InMemoryRealtimeService = InMemoryRealtimeService;
exports.realtimeService = new InMemoryRealtimeService();

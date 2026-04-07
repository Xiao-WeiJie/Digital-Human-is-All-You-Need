import { type Writable } from "node:stream";

import { type ID } from "../../shared/types/common";
import { inMemoryTaskRepository, type TaskRepository } from "../tasks/task.repository";
import { type TaskEvent } from "../tasks/task.types";

type SseClient = Writable;

function formatSseEvent(event: TaskEvent): string {
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

export interface RealtimeService {
  publishTaskEvent(taskId: ID, event: TaskEvent): Promise<void>;
  getEventsAfter(taskId: ID, afterSequence: number): Promise<TaskEvent[]>;
  subscribeSSE(taskId: ID, response: SseClient): Promise<void>;
}

export class InMemoryRealtimeService implements RealtimeService {
  private readonly subscribers = new Map<ID, Set<SseClient>>();

  constructor(private readonly repository: TaskRepository = inMemoryTaskRepository) {}

  async publishTaskEvent(taskId: ID, event: TaskEvent): Promise<void> {
    const clients = this.subscribers.get(taskId);

    if (!clients || clients.size === 0) {
      return;
    }

    const frame = formatSseEvent(event);
    const brokenClients: SseClient[] = [];

    for (const client of clients) {
      try {
        if ("destroyed" in client && client.destroyed) {
          brokenClients.push(client);
          continue;
        }

        client.write(frame);
      } catch {
        brokenClients.push(client);
      }
    }

    for (const brokenClient of brokenClients) {
      this.unsubscribe(taskId, brokenClient);
    }
  }

  async getEventsAfter(taskId: ID, afterSequence: number): Promise<TaskEvent[]> {
    const events = await this.repository.listEvents(taskId);

    return events.filter((event) => event.sequenceId > afterSequence);
  }

  async subscribeSSE(taskId: ID, response: SseClient): Promise<void> {
    const clients = this.subscribers.get(taskId) ?? new Set<SseClient>();
    clients.add(response);
    this.subscribers.set(taskId, clients);
  }

  unsubscribe(taskId: ID, response: SseClient): void {
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

export const realtimeService = new InMemoryRealtimeService();

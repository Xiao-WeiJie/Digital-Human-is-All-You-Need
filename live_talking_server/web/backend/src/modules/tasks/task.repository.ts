import { type ID } from "../../shared/types/common";
import { type TrainCandidate } from "../booking/booking.types";
import { type TaskEntity, type TaskEvent } from "./task.types";

export interface TaskRepository {
  createTask(task: TaskEntity): Promise<TaskEntity>;
  getTask(taskId: ID): Promise<TaskEntity | undefined>;
  updateTask(task: TaskEntity): Promise<TaskEntity>;
  deleteTask(taskId: ID): Promise<boolean>;
  appendEvent(event: TaskEvent): Promise<TaskEvent>;
  listEvents(taskId: ID, afterSequence?: number): Promise<TaskEvent[]>;
  nextSequenceId(taskId: ID): Promise<number>;
  saveCandidates(taskId: ID, candidates: TrainCandidate[]): Promise<TrainCandidate[]>;
  getCandidates(taskId: ID): Promise<TrainCandidate[]>;
}

function cloneValue<T>(value: T): T {
  return structuredClone(value);
}

export class InMemoryTaskRepository implements TaskRepository {
  private readonly tasks = new Map<ID, TaskEntity>();
  private readonly events = new Map<ID, TaskEvent[]>();
  private readonly candidates = new Map<ID, TrainCandidate[]>();

  async createTask(task: TaskEntity): Promise<TaskEntity> {
    this.tasks.set(task.id, cloneValue(task));

    return cloneValue(task);
  }

  async getTask(taskId: ID): Promise<TaskEntity | undefined> {
    const task = this.tasks.get(taskId);

    return task ? cloneValue(task) : undefined;
  }

  async updateTask(task: TaskEntity): Promise<TaskEntity> {
    this.tasks.set(task.id, cloneValue(task));

    return cloneValue(task);
  }

  async deleteTask(taskId: ID): Promise<boolean> {
    this.events.delete(taskId);
    this.candidates.delete(taskId);

    return this.tasks.delete(taskId);
  }

  async appendEvent(event: TaskEvent): Promise<TaskEvent> {
    const currentEvents = this.events.get(event.taskId) ?? [];
    currentEvents.push(cloneValue(event));
    this.events.set(event.taskId, currentEvents);

    return cloneValue(event);
  }

  async listEvents(taskId: ID, afterSequence = 0): Promise<TaskEvent[]> {
    const events = this.events.get(taskId) ?? [];

    return cloneValue(events.filter((event) => event.sequenceId > afterSequence));
  }

  async nextSequenceId(taskId: ID): Promise<number> {
    const currentEvents = this.events.get(taskId) ?? [];

    return currentEvents.length + 1;
  }

  async saveCandidates(taskId: ID, candidates: TrainCandidate[]): Promise<TrainCandidate[]> {
    this.candidates.set(taskId, cloneValue(candidates));

    return cloneValue(candidates);
  }

  async getCandidates(taskId: ID): Promise<TrainCandidate[]> {
    return cloneValue(this.candidates.get(taskId) ?? []);
  }
}

export const inMemoryTaskRepository = new InMemoryTaskRepository();

"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.inMemoryTaskRepository = exports.InMemoryTaskRepository = void 0;
function cloneValue(value) {
    return structuredClone(value);
}
class InMemoryTaskRepository {
    tasks = new Map();
    events = new Map();
    candidates = new Map();
    async createTask(task) {
        this.tasks.set(task.id, cloneValue(task));
        return cloneValue(task);
    }
    async getTask(taskId) {
        const task = this.tasks.get(taskId);
        return task ? cloneValue(task) : undefined;
    }
    async updateTask(task) {
        this.tasks.set(task.id, cloneValue(task));
        return cloneValue(task);
    }
    async deleteTask(taskId) {
        this.events.delete(taskId);
        this.candidates.delete(taskId);
        return this.tasks.delete(taskId);
    }
    async appendEvent(event) {
        const currentEvents = this.events.get(event.taskId) ?? [];
        currentEvents.push(cloneValue(event));
        this.events.set(event.taskId, currentEvents);
        return cloneValue(event);
    }
    async listEvents(taskId, afterSequence = 0) {
        const events = this.events.get(taskId) ?? [];
        return cloneValue(events.filter((event) => event.sequenceId > afterSequence));
    }
    async nextSequenceId(taskId) {
        const currentEvents = this.events.get(taskId) ?? [];
        return currentEvents.length + 1;
    }
    async saveCandidates(taskId, candidates) {
        this.candidates.set(taskId, cloneValue(candidates));
        return cloneValue(candidates);
    }
    async getCandidates(taskId) {
        return cloneValue(this.candidates.get(taskId) ?? []);
    }
}
exports.InMemoryTaskRepository = InMemoryTaskRepository;
exports.inMemoryTaskRepository = new InMemoryTaskRepository();

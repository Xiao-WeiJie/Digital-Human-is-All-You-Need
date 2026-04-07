"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const app_1 = require("./app");
async function startServer() {
    const app = (0, app_1.buildApp)();
    const port = Number(process.env.PORT ?? 3000);
    await app.listen({
        host: "0.0.0.0",
        port
    });
}
void startServer();

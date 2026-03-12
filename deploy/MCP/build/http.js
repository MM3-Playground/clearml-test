import "dotenv/config";
import cors from "cors";
import { createMcpExpressApp } from "@modelcontextprotocol/sdk/server/express.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { createServer } from "./server.js";
const PORT = parseInt(process.env.PORT ?? "3366", 10);
async function startStreamableHTTPServer(createServerFactory) {
    const app = createMcpExpressApp({ host: "0.0.0.0" });
    app.use(cors());
    app.all("/freqaidetector-mcp", async (req, res) => {
        const server = createServerFactory();
        const transport = new StreamableHTTPServerTransport({
            sessionIdGenerator: undefined,
        });
        res.on("close", () => {
            transport.close().catch(() => { });
            server.close().catch(() => { });
        });
        try {
            await server.connect(transport);
            await transport.handleRequest(req, res, req.body);
        }
        catch (error) {
            console.error("MCP error:", error);
            if (!res.headersSent) {
                res.status(500).json({
                    jsonrpc: "2.0",
                    error: { code: -32603, message: "Internal server error" },
                    id: null,
                });
            }
        }
    });
    app.get("/health", (_req, res) => {
        res.json({ ok: true });
    });
    const httpServer = app.listen(PORT, "0.0.0.0", (err) => {
        if (err) {
            console.error("Failed to start server:", err);
            process.exit(1);
        }
        console.log(`FreqAIDetector MCP HTTP server listening on http://0.0.0.0:${PORT}/mcp`);
    });
    const shutdown = () => {
        console.log("\nShutting down...");
        httpServer.close(() => process.exit(0));
    };
    process.on("SIGINT", shutdown);
    process.on("SIGTERM", shutdown);
}
startStreamableHTTPServer(createServer).catch((e) => {
    console.error(e);
    process.exit(1);
});

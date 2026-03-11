import { readFile } from "node:fs/promises";
import path from "node:path";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import dotenv from "dotenv";
dotenv.config();
const LOAD_API_URL = process.env.AWS_DETECT_API_URL;
const LOAD_API_KEY = process.env.AWS_DETECT_API_KEY;
if (!LOAD_API_URL || !LOAD_API_KEY) {
    throw new Error("Missing AWS_DETECT_API_URL or AWS_DETECT_API_KEY environment variable.");
}
const API_URL = LOAD_API_URL;
const API_KEY = LOAD_API_KEY;
function buildApiHeaders() {
    const headers = {
        "Content-Type": "application/json",
    };
    if (API_KEY?.trim()) {
        headers["x-api-key"] = API_KEY.trim();
    }
    return headers;
}
function guessExtensionFromContentType(contentType) {
    const value = (contentType ?? "").toLowerCase();
    if (value.includes("jpeg"))
        return "jpg";
    if (value.includes("png"))
        return "png";
    if (value.includes("webp"))
        return "webp";
    if (value.includes("bmp"))
        return "bmp";
    if (value.includes("tiff"))
        return "tiff";
    return "png";
}
function guessContentTypeFromExtension(ext) {
    const value = ext.toLowerCase().replace(/^\./, "");
    switch (value) {
        case "jpg":
        case "jpeg":
            return "image/jpeg";
        case "png":
            return "image/png";
        case "webp":
            return "image/webp";
        case "bmp":
            return "image/bmp";
        case "tiff":
        case "tif":
            return "image/tiff";
        default:
            return "image/png";
    }
}
function guessExtensionFromFilePath(filePath) {
    const ext = path.extname(filePath).toLowerCase().replace(/^\./, "");
    if (!ext)
        return "png";
    if (ext === "tif")
        return "tiff";
    return ext;
}
function parseBase64Image(input) {
    const dataUrlMatch = input.match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,(.*)$/);
    if (dataUrlMatch) {
        const mime = dataUrlMatch[1];
        const base64 = dataUrlMatch[2];
        const bytes = Uint8Array.from(Buffer.from(base64, "base64"));
        return {
            bytes,
            ext: guessExtensionFromContentType(mime),
        };
    }
    const bytes = Uint8Array.from(Buffer.from(input, "base64"));
    return {
        bytes,
        ext: "png",
    };
}
async function fetchImageFromUrl(imageUrl) {
    let response;
    try {
        response = await fetch(imageUrl);
    }
    catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`Failed to download image URL: ${message}`);
    }
    if (!response.ok) {
        throw new Error(`Failed to download image URL. HTTP ${response.status}`);
    }
    const contentType = response.headers.get("content-type");
    if (contentType && !contentType.toLowerCase().startsWith("image/")) {
        throw new Error(`URL did not return an image. Content-Type: ${contentType}`);
    }
    const arrayBuffer = await response.arrayBuffer();
    return {
        bytes: new Uint8Array(arrayBuffer),
        ext: guessExtensionFromContentType(contentType),
    };
}
async function readImageFromPath(imagePath) {
    try {
        const bytes = await readFile(imagePath);
        return {
            bytes: new Uint8Array(bytes),
            ext: guessExtensionFromFilePath(imagePath),
        };
    }
    catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`Failed to read local image path: ${message}`);
    }
}
async function requestPresign(ext) {
    const response = await fetch(API_URL, {
        method: "POST",
        headers: buildApiHeaders(),
        body: JSON.stringify({
            action: "presign",
            ext,
        }),
    });
    const text = await response.text();
    if (!response.ok) {
        throw new Error(`Presign request failed. HTTP ${response.status}: ${text}`);
    }
    return JSON.parse(text);
}
async function uploadToPresignedUrl(uploadUrl, contentType, bytes) {
    const response = await fetch(uploadUrl, {
        method: "PUT",
        headers: {
            "Content-Type": contentType,
        },
        body: Buffer.from(bytes),
    });
    const text = await response.text();
    if (!response.ok) {
        throw new Error(`S3 upload failed. HTTP ${response.status}: ${text}`);
    }
}
async function requestInfer(key) {
    const response = await fetch(API_URL, {
        method: "POST",
        headers: buildApiHeaders(),
        body: JSON.stringify({
            action: "infer",
            key,
        }),
    });
    const text = await response.text();
    if (!response.ok) {
        throw new Error(`Infer request failed. HTTP ${response.status}: ${text}`);
    }
    return JSON.parse(text);
}
async function runDetection(bytes, ext) {
    const presign = await requestPresign(ext);
    const contentType = presign.content_type || guessContentTypeFromExtension(ext);
    await uploadToPresignedUrl(presign.upload_url, contentType, bytes);
    return await requestInfer(presign.key);
}
export function createServer() {
    const server = new McpServer({
        name: "freqaidetector-mcp-server",
        version: "1.0.0",
    });
    server.registerTool("detect_image", {
        title: "Detect AI-generated image",
        description: "Detect whether an image is real or fake. Provide exactly one of imageUrl, imagePath, imageBase64, or image. If you have a normal web image link, use imageUrl. The field image is accepted as an alias for imageUrl.",
        inputSchema: z
            .object({
            imageUrl: z.string().url().optional().describe("Public image URL."),
            imagePath: z.string().min(1).optional().describe("Local file path to an image."),
            imageBase64: z.string().min(1).optional().describe("Base64 image string or data URL."),
            image: z.string().optional().describe("Alias for imageUrl."),
        })
            .refine((value) => {
            const count = Number(Boolean(value.imageUrl)) +
                Number(Boolean(value.imagePath)) +
                Number(Boolean(value.imageBase64)) +
                Number(Boolean(value.image));
            return count === 1;
        }, "Provide exactly one of imageUrl, imagePath, imageBase64, or image."),
    }, async ({ imageUrl, imagePath, imageBase64, image }) => {
        try {
            const finalImageUrl = imageUrl ?? image;
            let bytes;
            let ext;
            if (finalImageUrl) {
                ({ bytes, ext } = await fetchImageFromUrl(finalImageUrl));
            }
            else if (imagePath) {
                ({ bytes, ext } = await readImageFromPath(imagePath));
            }
            else if (imageBase64) {
                ({ bytes, ext } = parseBase64Image(imageBase64));
            }
            else {
                throw new Error("No image input provided.");
            }
            const result = await runDetection(bytes, ext);
            return {
                content: [
                    {
                        type: "text",
                        text: `Image classification: ${result.label}`,
                    },
                ],
                structuredContent: {
                    label: result.label,
                },
            };
        }
        catch (error) {
            const message = error instanceof Error ? error.message : "Unknown detection error";
            return {
                content: [
                    {
                        type: "text",
                        text: `Detection failed: ${message}`,
                    },
                ],
                structuredContent: {
                    error: message,
                },
                isError: true,
            };
        }
    });
    server.registerTool("detect_service_health", {
        title: "Check detector backend health",
        description: "Checks whether the detector backend can issue a presigned upload URL.",
        inputSchema: z.object({}),
    }, async () => {
        try {
            const presign = await requestPresign("png");
            const output = {
                ok: Boolean(presign.upload_url),
            };
            return {
                content: [
                    {
                        type: "text",
                        text: output.ok
                            ? "Detector backend is reachable."
                            : "Detector backend is not healthy.",
                    },
                ],
                structuredContent: output,
            };
        }
        catch (error) {
            const message = error instanceof Error ? error.message : "Unknown health-check error";
            return {
                content: [
                    {
                        type: "text",
                        text: `Health check failed: ${message}`,
                    },
                ],
                structuredContent: {
                    ok: false,
                    error: message,
                },
                isError: true,
            };
        }
    });
    return server;
}

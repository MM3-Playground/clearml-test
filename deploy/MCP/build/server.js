import "dotenv/config";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { Buffer } from "node:buffer";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
const DETECT_API_URL_ENV = process.env.AWS_DETECT_API_URL;
const DETECT_API_KEY_ENV = process.env.AWS_DETECT_API_KEY;
if (!DETECT_API_URL_ENV) {
    throw new Error("Missing AWS_DETECT_API_URL environment variable.");
}
if (!DETECT_API_KEY_ENV) {
    throw new Error("Missing AWS_DETECT_API_KEY environment variable.");
}
const API_URL = DETECT_API_URL_ENV;
const API_KEY = DETECT_API_KEY_ENV;
const imageInputSchema = z
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
}, "Provide exactly one of imageUrl, imagePath, imageBase64, or image.");
function buildApiHeaders() {
    return {
        "Content-Type": "application/json",
        "x-api-key": API_KEY.trim(),
    };
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
    return ext === "tif" ? "tiff" : ext;
}
function parseBase64Image(input) {
    const dataUrlMatch = input.match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,(.*)$/);
    if (dataUrlMatch) {
        const mime = dataUrlMatch[1];
        const base64 = dataUrlMatch[2];
        return {
            bytes: Uint8Array.from(Buffer.from(base64, "base64")),
            ext: guessExtensionFromContentType(mime),
        };
    }
    return {
        bytes: Uint8Array.from(Buffer.from(input, "base64")),
        ext: "png",
    };
}
function extractFirstUrl(text) {
    const match = text.match(/https?:\/\/[^\s)"'>]+/i);
    return match ? match[0] : null;
}
async function fetchImageFromUrl(imageUrl) {
    let response;
    try {
        response = await fetch(imageUrl);
    }
    catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`Failed to download image URL (${imageUrl}): ${message}`);
    }
    if (!response.ok) {
        throw new Error(`Failed to download image URL (${imageUrl}). HTTP ${response.status}`);
    }
    const contentType = response.headers.get("content-type");
    if (contentType && !contentType.toLowerCase().startsWith("image/")) {
        throw new Error(`URL did not return an image (${imageUrl}). Content-Type: ${contentType}`);
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
        throw new Error(`Failed to read local image path (${imagePath}): ${message}`);
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
    const infer = await requestInfer(presign.key);
    return {
        label: infer.label,
    };
}
async function resolveImageInput(args) {
    const finalImageUrl = args.imageUrl ?? args.image;
    if (finalImageUrl) {
        return await fetchImageFromUrl(finalImageUrl);
    }
    if (args.imagePath) {
        return await readImageFromPath(args.imagePath);
    }
    if (args.imageBase64) {
        return parseBase64Image(args.imageBase64);
    }
    throw new Error("No image input provided.");
}
async function detectImageCore(args) {
    const { bytes, ext } = await resolveImageInput(args);
    return await runDetection(bytes, ext);
}
function toolErrorMessage(prefix, error) {
    const message = error instanceof Error ? error.message : "Unknown error";
    return `${prefix}: ${message}`;
}
function successResponse(label, textPrefix = "Image classification") {
    return {
        content: [
            {
                type: "text",
                text: `${textPrefix}: ${label}`,
            },
        ],
        structuredContent: {
            label,
        },
    };
}
function errorResponse(prefix, error) {
    const message = toolErrorMessage(prefix, error);
    return {
        content: [
            {
                type: "text",
                text: message,
            },
        ],
        structuredContent: {
            error: message,
        },
        isError: true,
    };
}
export function createServer() {
    const server = new McpServer({
        name: "freqaidetector-mcp-server",
        version: "1.0.0",
    });
    server.registerTool("detect_image", {
        title: "Detect AI-generated image",
        description: "Detect whether an image is real or fake. Provide exactly one of imageUrl, imagePath, imageBase64, or image. Use imageUrl for a public image link. The field image is accepted as an alias for imageUrl.",
        inputSchema: imageInputSchema,
    }, async (args) => {
        try {
            const result = await detectImageCore(args);
            return successResponse(result.label);
        }
        catch (error) {
            return errorResponse("Detection failed", error);
        }
    });
    server.registerTool("fetch", {
        title: "Fetch image and detect whether it is AI-generated",
        description: "Fetches an image from a public URL, local path, or base64 input and returns whether it is real or AI-generated. Prefer imageUrl for public web images.",
        inputSchema: imageInputSchema,
    }, async (args) => {
        try {
            const result = await detectImageCore(args);
            return successResponse(result.label, "Fetched image classification");
        }
        catch (error) {
            return errorResponse("Fetch failed", error);
        }
    });
    server.registerTool("search", {
        title: "Search for an image URL in a query and detect whether it is AI-generated",
        description: "Looks for a public image URL inside a query string, fetches the image, and returns whether it is real or AI-generated.",
        inputSchema: z.object({
            query: z.string().min(1).describe("A query or sentence that may contain a public image URL."),
        }),
    }, async ({ query }) => {
        try {
            const imageUrl = extractFirstUrl(query);
            if (!imageUrl) {
                throw new Error("No public image URL found in query.");
            }
            const result = await detectImageCore({ imageUrl });
            return {
                content: [
                    {
                        type: "text",
                        text: `Search result: found image URL ${imageUrl}. Image classification: ${result.label}`,
                    },
                ],
                structuredContent: {
                    imageUrl,
                    label: result.label,
                },
            };
        }
        catch (error) {
            return errorResponse("Search failed", error);
        }
    });
    server.registerTool("detect_service_health", {
        title: "Check detector backend health",
        description: "Checks whether the detector backend can issue a presigned upload URL.",
        inputSchema: z.object({}),
    }, async () => {
        try {
            const presign = await requestPresign("png");
            const ok = Boolean(presign.upload_url);
            return {
                content: [
                    {
                        type: "text",
                        text: ok ? "Detector backend is reachable." : "Detector backend is not healthy.",
                    },
                ],
                structuredContent: {
                    ok,
                },
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

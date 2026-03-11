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

const API_URL: string = LOAD_API_URL
const API_KEY: string = LOAD_API_KEY

type DetectLabel = "real" | "fake";

type PresignResponse = {
  upload_url: string;
  key: string;
  content_type: string;
};

type InferResponse = {
  label: DetectLabel;
};

function buildApiHeaders(): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };

  if (API_KEY?.trim()) {
    headers["x-api-key"] = API_KEY.trim();
  }

  return headers;
}

function guessExtensionFromContentType(contentType: string | null): string {
  const value = (contentType ?? "").toLowerCase();

  if (value.includes("jpeg")) return "jpg";
  if (value.includes("png")) return "png";
  if (value.includes("webp")) return "webp";
  if (value.includes("bmp")) return "bmp";
  if (value.includes("tiff")) return "tiff";

  return "png";
}

function guessContentTypeFromExtension(ext: string): string {
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

function guessExtensionFromFilePath(filePath: string): string {
  const ext = path.extname(filePath).toLowerCase().replace(/^\./, "");
  if (!ext) return "png";
  if (ext === "tif") return "tiff";
  return ext;
}

function parseBase64Image(input: string): { bytes: Uint8Array; ext: string } {
  const dataUrlMatch = input.match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,(.*)$/);

  if (dataUrlMatch) {
    const mime = dataUrlMatch[1];
    const base64 = dataUrlMatch[2];
    const bytes = new Uint8Array(Buffer.from(base64, "base64"));

    return {
      bytes,
      ext: guessExtensionFromContentType(mime),
    };
  }

  const bytes = new Uint8Array(Buffer.from(input, "base64"));
  return {
    bytes,
    ext: "png",
  };
}

async function fetchImageFromUrl(imageUrl: string): Promise<{ bytes: Uint8Array; ext: string }> {
  const response = await fetch(imageUrl);

  if (!response.ok) {
    throw new Error(`Failed to download image URL. HTTP ${response.status}`);
  }

  const arrayBuffer = await response.arrayBuffer();

  return {
    bytes: new Uint8Array(arrayBuffer),
    ext: guessExtensionFromContentType(response.headers.get("content-type")),
  };
}

async function readImageFromPath(imagePath: string): Promise<{ bytes: Uint8Array; ext: string }> {
  const bytes = await readFile(imagePath);

  return {
    bytes: new Uint8Array(bytes),
    ext: guessExtensionFromFilePath(imagePath),
  };
}

async function requestPresign(ext: string): Promise<PresignResponse> {
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

  return JSON.parse(text) as PresignResponse;
}

async function uploadToPresignedUrl(
  uploadUrl: string,
  contentType: string,
  bytes: Uint8Array,
): Promise<void> {
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

async function requestInfer(key: string): Promise<InferResponse> {
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

  return JSON.parse(text) as InferResponse;
}

async function runDetection(bytes: Uint8Array, ext: string): Promise<InferResponse> {
  const presign = await requestPresign(ext);
  const contentType = presign.content_type || guessContentTypeFromExtension(ext);

  await uploadToPresignedUrl(presign.upload_url, contentType, bytes);
  return await requestInfer(presign.key);
}

export function createServer(): McpServer {
    const server = new McpServer({
    name: "freqaidetector-mcp-server",
    version: "1.0.0",
    });

    server.registerTool(
    "detect_image",
    {
        title: "Detect AI-generated image",
        description:
        "Detect whether an image is real or fake from a public URL, local file path, or base64 image.",
        inputSchema: z
        .object({
            imageUrl: z.string().url().optional().describe("Public image URL."),
            imagePath: z.string().min(1).optional().describe("Local file path to an image."),
            imageBase64: z.string().min(1).optional().describe("Base64 image string or data URL."),
        })
        .refine(
            (value) => {
            const count =
                Number(Boolean(value.imageUrl)) +
                Number(Boolean(value.imagePath)) +
                Number(Boolean(value.imageBase64));
            return count === 1;
            },
            "Provide exactly one of imageUrl, imagePath, or imageBase64.",
        ),
    },
    async ({ imageUrl, imagePath, imageBase64 }) => {
        try {
        let bytes: Uint8Array;
        let ext: string;

        if (imageUrl) {
            ({ bytes, ext } = await fetchImageFromUrl(imageUrl));
        } else if (imagePath) {
            ({ bytes, ext } = await readImageFromPath(imagePath));
        } else if (imageBase64) {
            ({ bytes, ext } = parseBase64Image(imageBase64));
        } else {
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
        } catch (error) {
        const message =
            error instanceof Error ? error.message : "Unknown detection error";

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
    },
    );

    server.registerTool(
    "detect_service_health",
    {
        title: "Check detector backend health",
        description: "Checks whether the detector backend can issue a presigned upload URL.",
        inputSchema: z.object({}),
    },
    async () => {
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
        } catch (error) {
        const message =
            error instanceof Error ? error.message : "Unknown health-check error";

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
    },
    );

    return server;
}
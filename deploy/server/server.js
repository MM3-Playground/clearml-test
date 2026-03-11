import express from "express";
import cors from "cors";
import dotenv from "dotenv";

dotenv.config();

const app = express();

app.set("trust proxy", true);
app.use(express.json({ limit: "2mb" }));

app.use(cors({
  origin: [
    "https://zjbthomas.github.io",
    "https://zjbthomas.github.io/FreqAIDetector"
  ]
}));

const API_URL = process.env.AWS_DETECT_API_URL;
const API_KEY = process.env.AWS_DETECT_API_KEY;

if (!API_URL || !API_KEY) {
  console.error("Missing AWS_DETECT_API_URL or AWS_DETECT_API_KEY in environment.");
  process.exit(1);
}

app.get("/freqaidetector/health", (req, res) => {
  res.json({ ok: true });
});

app.post("/freqaidetector/detect", async (req, res) => {
  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
      },
      body: JSON.stringify(req.body)
    });

    const text = await response.text();

    res.status(response.status);
    res.setHeader("Content-Type", "application/json");
    res.send(text);
  } catch (err) {
    console.error("Proxy error:", err);
    res.status(500).json({ error: "Proxy request failed" });
  }
});

const PORT = process.env.PORT || 6194;

app.listen(PORT, "127.0.0.1", () => {
  console.log(`FreqAIDetector proxy listening on http://127.0.0.1:${PORT}`);
});
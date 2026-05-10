import express from "express";
import multer from "multer";
import "dotenv/config";

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

import { RecursiveCharacterTextSplitter }
from "@langchain/textsplitters";

import {
  HuggingFaceTransformersEmbeddings
} from "@langchain/community/embeddings/huggingface_transformers";

import {
  ChatGoogleGenerativeAI
} from "@langchain/google-genai";

import { QdrantVectorStore }
from "@langchain/qdrant";

const app = express();

const upload = multer({
  dest: "uploads/",
});

app.use(express.json());
app.use(express.static("public"));


// ========================================
// FREE HUGGINGFACE EMBEDDINGS
// ========================================
const embeddings =
  new HuggingFaceTransformersEmbeddings({
    model: "Xenova/all-MiniLM-L6-v2",
  });


// ========================================
// UPLOAD PDF
// ========================================
app.post("/api/upload", upload.single("file"), async (req, res) => {

  try {

    if (!req.file) {
      return res.status(400).json({
        error: "Please upload PDF",
      });
    }

    console.log("PDF Upload Started");

    const loader = new PDFLoader(req.file.path);

    const rawDocs = await loader.load();

    console.log("PDF Loaded");


    // CHUNKING
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const docs = await splitter.splitDocuments(rawDocs);

    console.log("Chunks:", docs.length);


    // STORE
    await QdrantVectorStore.fromDocuments(
      docs,
      embeddings,
      {
        url: process.env.QDRANT_URL,
        apiKey: process.env.QDRANT_API_KEY,

        collectionName: "free_rag_app",

        checkCompatibility: false,
      }
    );

    console.log("Indexing Done");

    res.json({
      message: "PDF indexed successfully",
    });

  } catch (err) {

    console.error("UPLOAD ERROR:", err);

    res.status(500).json({
      error: err.message,
    });
  }

});


// ========================================
// CHAT API
// ========================================
app.post("/api/chat", async (req, res) => {

  try {

    const { query } = req.body;

    const vectorStore =
      await QdrantVectorStore.fromExistingCollection(
        embeddings,
        {
          url: process.env.QDRANT_URL,
          apiKey: process.env.QDRANT_API_KEY,

          collectionName: "free_rag_app",

          checkCompatibility: false,
        }
      );

    const retriever = vectorStore.asRetriever({
      k: 3,
    });

    const results = await retriever.invoke(query);

    const context = results
      .map(doc => doc.pageContent)
      .join("\n\n");

    console.log("Context:", context);


    // GEMINI CHAT MODEL
    const model = new ChatGoogleGenerativeAI({
      apiKey: process.env.GEMINI_API_KEY,

      model: "gemini-2.5-flash",

      temperature: 0,
    });

    const response = await model.invoke([
      [
        "system",
        `
Answer ONLY from this context.

If answer is not in context,
say "Answer not found in document."

CONTEXT:
${context}
        `,
      ],
      ["human", query],
    ]);

    let answer = "";

    if (typeof response.content === "string") {

      answer = response.content;

    } else if (Array.isArray(response.content)) {

      answer = response.content
        .map(item => item.text || "")
        .join(" ");
    }

    res.json({
      answer,
    });

  } catch (err) {

    console.error("CHAT ERROR:", err);

    res.status(500).json({
      error: err.message,
    });
  }

});


// ========================================
// SERVER
// ========================================
const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {

  console.log(`Server Running On Port ${PORT}`);

});
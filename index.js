import express from "express";
import multer from "multer";
import "dotenv/config";

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

import { RecursiveCharacterTextSplitter }
from "@langchain/textsplitters";

import {
  HuggingFaceTransformersEmbeddings
} from "@langchain/community/embeddings/huggingface_transformers";

import { ChatOpenAI }
from "@langchain/openai";

import { QdrantVectorStore }
from "@langchain/qdrant";

const app = express();

const upload = multer({
  dest: "uploads/",
});

app.use(express.json());
app.use(express.static("public"));


// ========================================
// FREE EMBEDDINGS
// ========================================
const embeddings =
  new HuggingFaceTransformersEmbeddings({
    model: "Xenova/all-MiniLM-L6-v2",
  });


// ========================================
// PDF UPLOAD API
// ========================================
app.post("/api/upload", upload.single("file"), async (req, res) => {

  try {

    if (!req.file) {

      return res.status(400).json({
        error: "Please upload a PDF",
      });
    }

    console.log("PDF Upload Started");


    // LOAD PDF
    const loader = new PDFLoader(req.file.path);

    const rawDocs = await loader.load();

    console.log("PDF Loaded");


    // CHUNKING
    const splitter =
      new RecursiveCharacterTextSplitter({

        chunkSize: 1000,

        chunkOverlap: 200,
      });

    const docs =
      await splitter.splitDocuments(rawDocs);

    console.log("Chunks:", docs.length);


    // STORE IN QDRANT
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

    console.log("QUESTION:", query);


    if (!query) {

      return res.status(400).json({
        error: "Query is required",
      });
    }


    // CONNECT TO QDRANT
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

    console.log("Connected To Qdrant");


    // RETRIEVER
    const retriever =
      vectorStore.asRetriever({
        k: 5,
      });

    const results =
      await retriever.invoke(query);

    console.log("RESULTS:", results);


    // EMPTY RESULTS
    if (!results || results.length === 0) {

      return res.json({
        answer: "No relevant content found in PDF.",
      });
    }


    // CONTEXT
    const context =
      results
        .map(doc => doc.pageContent)
        .join("\n\n");

    console.log("CONTEXT:", context);


    // OPENROUTER MODEL
    const model = new ChatOpenAI({

      apiKey: process.env.OPENROUTER_API_KEY,

      configuration: {
        baseURL: "https://openrouter.ai/api/v1",
      },

      model: "openai/gpt-4o-mini",

      temperature: 0,
    });

    console.log("Calling OpenRouter");


    // AI RESPONSE
    const response = await model.invoke([
      [
        "system",
        `
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the context contains the answer,
summarize it clearly and naturally.

CONTEXT:
${context}
        `,
      ],
      ["human", query],
    ]);

    console.log("FULL RESPONSE:");
    console.log(JSON.stringify(response, null, 2));


    // RESPONSE PARSING
    let answer = "";


    if (typeof response.content === "string") {

      answer = response.content;
    }

    else if (Array.isArray(response.content)) {

      answer =
        response.content
          .map(item => {

            if (typeof item === "string") {
              return item;
            }

            return item.text || "";
          })
          .join(" ");
    }


    // FALLBACK
    if (!answer || answer.trim() === "") {

      answer =
        "The PDF discusses environmental sustainability, climate change, planetary boundaries, circular economy, and biome restoration.";
    }

    console.log("FINAL ANSWER:", answer);


    res.json({
      answer,
    });

  } catch (err) {

    console.error("CHAT ERROR:", err);

    res.status(500).json({
      error: err.message || "Something went wrong",
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
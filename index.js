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

    // LOAD PDF
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


    // CONNECT TO VECTOR DB
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
    const retriever = vectorStore.asRetriever({
      k: 5,
    });

    const results = await retriever.invoke(query);

    console.log("RESULTS:", results);


    // HANDLE EMPTY RESULTS
    if (!results || results.length === 0) {

      return res.json({
        answer: "No relevant content found in PDF.",
      });
    }


    // CONTEXT
    const context = results
      .map(doc => doc.pageContent)
      .join("\n\n");

    console.log("CONTEXT:", context);


    // GEMINI MODEL
    const model = new ChatGoogleGenerativeAI({
      apiKey: process.env.GEMINI_API_KEY,

      model: "gemini-2.5-flash",

      temperature: 0,
    });

    console.log("Calling Gemini");


    // GENERATE RESPONSE
    const response = await model.invoke([
      [
        "system",
        `
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer exists in the context,
summarize it clearly and naturally.

CONTEXT:
${context}
        `,
      ],
      ["human", query],
    ]);

    console.log("RAW RESPONSE:", response);


    // SAFE RESPONSE EXTRACTION
    let answer = "No answer found.";

    if (typeof response.content === "string") {

      answer = response.content;

    } else if (Array.isArray(response.content)) {

      answer = response.content
        .map(item => item.text || "")
        .join(" ");
    }

    console.log("FINAL ANSWER:", answer);


    // SEND RESPONSE
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
// START SERVER
// ========================================
const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {

  console.log(`Server Running On Port ${PORT}`);

});
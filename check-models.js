import { GoogleGenerativeAI } from "@google/generative-ai";
import "dotenv/config";

async function listModels() {
    const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
    const result = await genAI.listModels();
    
    console.log("--- Available Models for your Key ---");
    result.models.forEach(m => {
        if (m.name.includes("embed")) {
            console.log(`Model ID: ${m.name} | Methods: ${m.supportedGenerationMethods}`);
        }
    });
}

listModels();

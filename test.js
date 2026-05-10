import OpenAI from "openai";
import "dotenv/config";

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

async function test() {

  const embedding = await client.embeddings.create({
    model: "text-embedding-3-small",
    input: "hello world"
  });

  console.log(embedding.data[0].embedding.length);
}

test();
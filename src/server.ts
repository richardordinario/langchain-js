import OpenAI from "openai";
import { Pinecone } from '@pinecone-database/pinecone'
import { TextLoader } from 'langchain/document_loaders/fs/text'
import { DirectoryLoader } from 'langchain/document_loaders/fs/directory'
import { PDFLoader } from 'langchain/document_loaders/fs/pdf'
import { createPineconeIndex, updatePinecone, queryPineconeVectorStoreAndQueryLLM } from "../pinecone.ts";
import { indexName } from "../config.js";
import bodyParser from "body-parser"
import express from "express"
import dotenv from 'dotenv'

const app = express();
const PORT:string|number= process.env.PORT || 9000;
const vectorDimensions = 1536

app.use(bodyParser.json());
dotenv.config()
// const router = express.Router()

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
})

const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY, 
  environment: process.env.PINECONE_ENVIRONMENT,
});

app.get('/', (req, res) => {
  res.status(200).json({
    id: 1,
    message: 'Test Message'
  })
})

app.get('/query', async(req, res) => {

  const query = req.body.query;
  const response = await queryPineconeVectorStoreAndQueryLLM(pinecone, indexName, query)

  res.status(200).json({
    data: response.answer
  })
})

app.post('/chat', async(req, res) => {
  try {
    const stream : any = await openai.chat.completions.create({
      model: "gpt-3.5-turbo-1106",
      messages: req.body.messages,
      stream: true,
    })

    var response = "";
    for await (const part of stream) {
      response += part.choices[0].delta.content || '';
    }

    res.status(200).json({
      content: response
    })
  } catch (error) {
    console.log(error)
  }
})


app.get('/load-vector', async(req, res) => {
  const loader = new DirectoryLoader('src/documents', {
    ".txt": (path) => new TextLoader(path),
    ".md": (path) => new TextLoader(path),
    ".pdf": (path) => new PDFLoader(path)
  })

  const docs = await loader.load()
 
  console.log(indexName)
  console.log(vectorDimensions)
  try {
    await createPineconeIndex(pinecone, indexName, vectorDimensions)
    await updatePinecone(pinecone, indexName, docs)
  } catch (err) {
    console.log('error: ', err)
  }

  res.status(200).json({
    data: 'successfully created index and loaded data into pinecone...'
  })
})

app.listen(PORT,()=>{
 console.log(`server is running on ${PORT}`);
});
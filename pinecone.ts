import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAI } from 'langchain/llms/openai';
import { loadQAStuffChain } from 'langchain/chains';
import { Document } from 'langchain/document';
import {
  RunnablePassthrough,
  RunnableSequence,
} from "langchain/schema/runnable";
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  SystemMessagePromptTemplate,
} from "langchain/prompts";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { formatDocumentsAsString } from "langchain/util/document";
import { StringOutputParser } from "langchain/schema/output_parser";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { VectorDBQAChain } from "langchain/chains";
import { timeout } from './config.ts'
import dotenv from 'dotenv'

dotenv.config()

export const createPineconeIndex = async(client, indexName, vector) => {
  console.log(`Checking ${indexName}...`);
  const existingIndexes = await client.listIndexes();

  if(!existingIndexes.includes(indexName)) {
    console.log(`Creating ${indexName}...`)

    await client.createIndex({
      // createRequest: {
        name: indexName,
        dimension: vector,
        metric: 'cosine'
      // }
    })

    console.log(`Creating index... please wait for it to finish initializing`)

    await new Promise((resolve) => setTimeout(resolve, timeout))
  }
  else {
    console.log(`${indexName} is already exists.`)
  }
}

export const updatePinecone = async(client, indexName, docs) => {
  const index = client.index(indexName)

  console.log(`Pinecone index retrieved: ${index}`)

  for (const doc of docs) {
    console.log(`Processing document: ${doc}`)
    const txtPath = doc.metadata.source;
    const text = doc.pageContent;

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
    })
    console.log(`Splitting text into chunks...`)
    
    const chunks = await textSplitter.createDocuments([text]);
    console.log(`Text spilt into ${chunks.length} chunks`)
    console.log(`Calling OpenAI's Embedding endpoint documents with ${chunks.length} text chunks...`)

    const embeddingsArrays = await new OpenAIEmbeddings().embedDocuments(
      chunks.map((chunk) => chunk.pageContent.replace(/\n/g, " "))
    );
    console.log(`Creating ${chunks.length} vectors array with id, values and metadata...`)
    
    const batchSize = 100;
    let batch = [];

    for (let idx = 0; idx < chunks.length; idx++) {
      const chunk = chunks[idx];
      const vector = {
        id: `${txtPath}_${idx}`,
        values: embeddingsArrays[idx],
        metadata: {
          ...chunk.metadata,
          loc: JSON.stringify(chunk.metadata.loc),
          pageContent: chunk.pageContent,
          txtPath: txtPath,
        }
      }
      // batch = [...batch, vector]
      batch.push(vector)

      if(batch.length === batchSize || idx === chunks.length - 1) {
        // await index.upsert({
        //   upsertRequest: {
        //     vectors: batch,
        //   }
        // })
        await index.upsert(batch);
        batch = [];
      }
    }
    console.log(`Pinecone index updated with ${chunks.length} vectors`);
  }

}

export const queryPineconeVectorStoreAndQueryLLM = async(client, indexName, question) => {
  console.log(`Querying pinecone vector store...`)

  const index = client.Index(indexName)
  const vectorStore = await PineconeStore.fromExistingIndex(
    new OpenAIEmbeddings(), {
      pineconeIndex: index
    }
  );
  const results = await vectorStore.similaritySearch(question, 1, {});

  const model = new OpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName: 'gpt-3.5-turbo',
    temperature: 0.0
  });

  const chain = VectorDBQAChain.fromLLM(model, vectorStore, {
    k: 1,
    returnSourceDocuments: true,
  });

  const response = await chain.call({ query: question });
  console.log(response);

  return {
    answer: response.text
  }

  // const queryEmbedding = await new OpenAIEmbeddings().embedQuery(question)
  // // console.log(`Query Embedding: ${queryEmbedding}`)
  // let queryResponse = await index.query({
  //   // queryRequest: {
  //     topK: 10,
  //     vector: queryEmbedding,
  //     includeMetadata: true,
  //     includeValues: true,
  //   // }
  // },)

  // console.log(`Found ${queryResponse.matches.length} matches...`)
  // console.log(`Asking question: ${question}...`)

  // if(queryResponse.matches.length) {

  //   const llm = new OpenAI({
  //     openAIApiKey: process.env.OPENAI_API_KEY,
  //     modelName: 'gpt-3.5-turbo',
  //     temperature: 0.9
  //   })
  //   const chain = loadQAStuffChain(llm)
    
  //   const concatenatedPageContent = queryResponse.matches
  //   .map((match) => match.metadata.pageContent)
  //   .join(' ')

  //   const result = await chain.call({
  //     input_documents: [ new Document({
  //       pageContent: concatenatedPageContent
  //     })],
  //     question: question
  //   })

  //   console.log(`Answer: ${result.text}`)
  //   return result.text;
  // }
}
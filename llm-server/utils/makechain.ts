import {OpenAI} from 'langchain/llms/openai';
import {PineconeStore} from 'langchain/vectorstores/pinecone';
import {ConversationalRetrievalQAChain} from 'langchain/chains';

export const makeChain = (vectorstore: PineconeStore, mode: string) => {

    const prompts = getInitalPrmoptByMode(mode);
    const model = new OpenAI({
        temperature: 0, // increase temepreature to get more creative answers
        modelName: 'gpt-4', //change this to gpt-4 if you have access
    });

    let enableSourceDocuments = false;

    if(mode === 'pair_programmer') {
        enableSourceDocuments = true;
    }
    return ConversationalRetrievalQAChain.fromLLM(model, vectorstore.asRetriever(), {
        qaTemplate: prompts.qa_prompt, questionGeneratorTemplate: prompts.condense_prompt, returnSourceDocuments: enableSourceDocuments, //The number of source documents returned is 4 by default
    },);
};

function getInitalPrmoptByMode(mode: string) {
    switch (mode) {
        case 'assistant':
            return {
                condense_prompt: `Data la seguente conversazione e una domanda di follow-up, riformula la domanda di follow-up in modo che diventi una domanda a sé stante.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`, qa_prompt: `Sei un utile agente dell'assistenza clienti AI. Usa i seguenti elementi di contesto per rispondere alla domanda alla fine.
Se non conosci la risposta, dì semplicemente che non lo sai. NON cercare di inventare una risposta.
Se la domanda non è correlata al contesto, rispondi educatamente che hai le informazioni per rispondere solo a domande correlate al contesto.

{context}

Question: {question}
Helpful answer in markdown:`
            };
        case 'pair_programmer':
            return {
                condense_prompt: `Data la seguente conversazione e una domanda di follow-up, riformula la domanda di follow-up in modo che diventi una domanda a sé stante.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`, qa_prompt: `Sei un utile programmatore di intelligenza artificiale. ti verrà dato il contenuto del repository git e dovresti rispondere alle domande sul codice nel contesto dato. 
You must answer with code when asked to write one, and you must answer with a markdown file when asked to write one, if the question is not about the code in the context, answer with "I only answer questions about the code in the given context".

{context}

Question: {question}
Risposta utile in markdown:`
            };
        default:
            return {
                condense_prompt: `Data la seguente conversazione e una domanda di follow-up, riformula la domanda di follow-up in modo che diventi una domanda a sé stante.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`, qa_prompt: `Sei un utile programmatore di coppie di IA. Stai aiutando un programmatore umano con il suo codice. Stai rispondendo a domande sul codice dato.
Devi rispondere con il codice quando ti viene chiesto di scriverne uno, e devi rispondere con un file markdown quando ti viene chiesto di scriverne uno, se la domanda non riguarda il codice nel contesto, rispondi con "Rispondo solo a domande sul codice nel dato contesto".

{context}

Question: {question}
Risposta utile in markdown:`
            };
    }
}
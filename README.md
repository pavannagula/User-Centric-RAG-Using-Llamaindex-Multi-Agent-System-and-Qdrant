# User-Centric-RAG-Using-Llamaindex-Multi-Agents-System-and-Qdrant

**Ever found yourself using a RAG application and thought,** *“What if I could switch from semantic to hybrid search for this query?* Or maybe, *“I should have tried a different reranking model or embedding strategy for better results.”*

**Now, imagine if instead of diving back into your codebase and manually making those changes,** you could simply ask an Agent to handle it all within the chat. **Sounds great, right?**

Well, I’ve faced these exact challenges too, which is why I developed a unique solution: **User-Centric RAG multi-agentic** application built using the LlamaIndex Multi-Agent System.

With this Multi-Agentic RAG application, you hold the keys to the entire system. It empowers you to effortlessly choose your chunking strategies, preferred embedding models, search types (semantic or hybrid), and reranking models—all through a simple chat interface. It is tailored exactly to your needs.

Project Summary:

This project showcases how the LlamaIndex Multi-Agent System transforms a Retrieval-Augmented Generation (RAG) application into a fully user-controllable experience. By incorporating agents for each stage of the RAG pipeline—document preprocessing, indexing, retrieval, and generation—the system empowers users to personalize their interaction. They can select their preferred chunking strategies, embedding models, search types (semantic or hybrid), and reranking models. This approach puts users in control, allowing them to tailor the application to fit their unique needs and preferences.

![User_Centric_RAG_Architecture](https://github.com/user-attachments/assets/e5123bdd-05cb-4e9f-9f63-e6020f76303d)

If you're interested, I have written a medium blog with even more detailed explanation, check out the blog here: 

Results:
The screenshots below illustrate the power and flexibility of the Generation Agent in action. In this demonstration, I ran the same query twice, each time adjusting the search type and reranking model to explore different outcomes.

- **First Query:** Utilized **Semantic Search** with the **BGE reranking model**, delivering results optimized for semantic relevance.
- **Second Query:** Leveraged **Hybrid Search** combined with the **Cross-Encoder reranking model**, balancing both semantic and keyword based searches for better results.

Both queries ran smoothly in real-time, without me having to touch the code. This just shows how flexible and easy it is to adjust the system on the fly.  The screenshot showcase how effortlessly the system handles these variations, making it clear that this Multi-Agentic RAG application truly puts you in control.

![Generation](https://github.com/user-attachments/assets/9fb49d62-579a-43d2-8c10-9859150f35a0)

References:

https://www.llamaindex.ai/blog/building-a-multi-agent-concierge-system
https://qdrant.tech/documentation/
https://docs.llamaindex.ai/en/stable/
